"""Decoder selection, checkpoint compatibility, and complete-model fusion.

Uses synthetic tensors and random initialization; no dataset, pretrained weights,
or training job is required. Checkpoint tests use the real BaseModel heads on a
small feature tensor so that they do not write full DLA checkpoints.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src/lib")]

from opts import opts
from model.model import create_model, load_model, save_model
from model.networks.base_model import BaseModel, QAttention
from model.utils import fuse_model


def options(use_bfl=False, arch="dla34", pre_hm=True, extra=()):
    parser = opts()
    argv = ["--arch", arch, "--no_pretrained", "--device", "cpu",
            "--local_rank", "1", "--input_h", "64", "--input_w", "64"]
    if use_bfl:
        argv.append("--use_bfl")
    if pre_hm:
        argv.append("--pre_hm")
    return parser.update_res_info_and_set_heads(parser.parse(argv + list(extra)))


class FeatureModel(BaseModel):
    """Exercise production decoder construction without a DLA checkpoint."""

    def __init__(self, opt):
        super().__init__(opt.heads, {name: [8] for name in opt.heads},
                         num_stacks=1, last_channel=4, opt=opt)
        self.opt = opt

    def img2feats(self, image):
        return [image]


class TrainingModel(nn.Module):
    """Keep the model./loss. namespaces used by production training saves."""

    def __init__(self, opt):
        super().__init__()
        self.model = FeatureModel(opt)
        self.loss = nn.Linear(4, 2)


class DecoderModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(317)

    def assert_outputs_close(self, expected, actual, rtol=1e-4, atol=1e-5):
        self.assertEqual(len(expected), len(actual))
        for before, after in zip(expected, actual):
            self.assertEqual(set(before), set(after))
            for name in before:
                self.assertTrue(torch.isfinite(after[name]).all(), name)
                torch.testing.assert_close(before[name], after[name], rtol=rtol, atol=atol)

    def assert_states_equal(self, expected, actual):
        self.assertEqual(set(expected), set(actual))
        for name in expected:
            torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)

    def test_parser_defaults_and_explicit_main_method(self):
        baseline = options()
        main = options(use_bfl=True)
        self.assertFalse(baseline.use_bfl)
        self.assertTrue(main.use_bfl)
        self.assertEqual(baseline.head_kernel, 3)
        self.assertEqual(baseline.heads, main.heads)
        self.assertEqual(baseline.head_conv, main.head_conv)
        self.assertEqual(baseline.weights, main.weights)
        self.assertEqual(set(main.heads), {"hm", "reg", "tracking", "ltrb_amodal", "reid"})

    def test_head_kernel_changes_only_the_conv_baseline(self):
        for use_bfl in (False, True):
            with self.subTest(use_bfl=use_bfl):
                model = FeatureModel(options(use_bfl=use_bfl, extra=["--head_kernel", "5"]))
                for name in model.heads:
                    head = getattr(model, name)
                    self.assertIsInstance(head[1], nn.ReLU)
                    self.assertIsInstance(head[2], nn.Conv2d)
                    self.assertEqual(head[2].kernel_size, (1, 1))
                    if use_bfl:
                        self.assertIsInstance(head[0], QAttention)
                        for projection in (head[0].conv1, head[0].conv2,
                                           head[0].conv3, head[0].conv4):
                            self.assertEqual(projection.kernel_size, (1, 1))
                    else:
                        self.assertIs(type(head[0]), nn.Conv2d)
                        self.assertEqual(head[0].kernel_size, (5, 5))
                        self.assertEqual(head[0].padding, (2, 2))
                        self.assertIsNotNone(head[0].bias)
                        self.assertFalse(any(isinstance(m, nn.BatchNorm2d) for m in head.modules()))

    def test_qattention_preserves_legacy_checkpoint_parameter_names(self):
        model = FeatureModel(options(use_bfl=True))
        expected_first_layer = {
            "conv1.weight", "conv2.weight", "conv3.weight", "conv4.weight", "conv4.bias",
            "bn4.weight", "bn4.bias", "bn4.running_mean", "bn4.running_var",
            "bn4.num_batches_tracked",
        }
        for name in model.heads:
            with self.subTest(head=name):
                keys = {key[len(name) + 3:] for key in model.state_dict()
                        if key.startswith(name + ".0.")}
                self.assertEqual(keys, expected_first_layer)
                self.assertIn(name + ".2.weight", model.state_dict())
                self.assertIn(name + ".2.bias", model.state_dict())

    def _exercise_model(self, arch, use_bfl, pre_hm, backward):
        opt = options(use_bfl=use_bfl, arch=arch, pre_hm=pre_hm)
        model = create_model(opt.arch, opt.heads, opt.head_conv, opt)
        input_channels = 64 if arch == "dla34" else 128
        first_layers = []
        for name, channels in opt.heads.items():
            head = getattr(model, name)
            first_layers.append(head[0])
            if use_bfl:
                self.assertIsInstance(head[0], QAttention)
                self.assertEqual(head[0].conv1.in_channels, input_channels)
            else:
                self.assertIs(type(head[0]), nn.Conv2d)
                self.assertEqual(head[0].in_channels, input_channels)
                self.assertEqual(head[0].kernel_size, (3, 3))
                self.assertIsNotNone(head[0].bias)
            if backward:
                features = torch.randn(2, input_channels, 4, 4, requires_grad=True)
                prediction = head(features)
                self.assertEqual(prediction.shape, (2, channels, 4, 4))
                prediction.square().mean().backward()
                self.assertIsNotNone(features.grad)
                self.assertTrue(torch.isfinite(features.grad).all(), name)
                for parameter in head.parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                head.zero_grad(set_to_none=True)
        self.assertEqual(len({id(layer) for layer in first_layers}), len(opt.heads))

        model.eval()
        current = torch.randn(1, 3, 64, 64)
        previous = torch.randn_like(current)
        heatmap = torch.rand(1, 1, 64, 64) if pre_hm else None
        self.assertEqual(hasattr(model.base, "pre_hm_layer"), pre_hm)
        with torch.no_grad():
            expected = model(current, previous, heatmap)
            self.assertEqual(set(expected[0]), set(opt.heads))
            for name, channels in opt.heads.items():
                self.assertEqual(expected[0][name].shape, (1, channels, 16, 16))
                self.assertTrue(torch.isfinite(expected[0][name]).all(), name)
            # In-place fusion avoids retaining a second large DLA model.
            fused = fuse_model(model)
            actual = fused(current, previous, heatmap)
        self.assert_outputs_close(expected, actual)

    def test_dla34_and_dla60_modes_forward_head_backward_and_fusion(self):
        for arch in ("dla34", "dla60"):
            for use_bfl in (False, True):
                with self.subTest(arch=arch, use_bfl=use_bfl):
                    self._exercise_model(arch, use_bfl, pre_hm=True, backward=True)

    def test_fusion_with_no_previous_heatmap_stem(self):
        for use_bfl in (False, True):
            with self.subTest(use_bfl=use_bfl):
                self._exercise_model("dla34", use_bfl, pre_hm=False, backward=False)

    def test_each_mode_strict_checkpoint_round_trip_and_cross_mode_rejection(self):
        features = torch.randn(2, 4, 6, 6)
        with tempfile.TemporaryDirectory(prefix="gatemot_decoder_checkpoint_") as temp:
            for use_bfl in (False, True):
                with self.subTest(use_bfl=use_bfl):
                    opt = options(use_bfl=use_bfl)
                    source = FeatureModel(opt).eval()
                    checkpoint = str(Path(temp) / ("qattn.pth" if use_bfl else "conv.pth"))
                    save_model(checkpoint, 3, source)
                    restored = FeatureModel(opt).eval()
                    load_model(restored, checkpoint, opt)
                    self.assert_states_equal(source.state_dict(), restored.state_dict())
                    with torch.no_grad():
                        self.assert_outputs_close(source(features), restored(features), rtol=0, atol=0)

                    other_opt = options(use_bfl=not use_bfl)
                    other = FeatureModel(other_opt)
                    before = {name: value.clone() for name, value in other.state_dict().items()}
                    with self.assertRaisesRegex(RuntimeError, "(?i)mismatch"):
                        load_model(other, checkpoint, other_opt)
                    self.assert_states_equal(before, other.state_dict())

                    # Evaluation may never silently initialize unmatched heads.
                    opt.allow_partial_load = True
                    with self.assertRaises((ValueError, RuntimeError)):
                        load_model(restored, checkpoint, opt)

    def test_training_resume_restores_matching_mode_and_rejects_partial_or_cross_mode(self):
        features = torch.randn(2, 4, 6, 6)
        with tempfile.TemporaryDirectory(prefix="gatemot_decoder_resume_") as temp:
            for use_bfl in (False, True):
                with self.subTest(use_bfl=use_bfl):
                    opt = options(use_bfl=use_bfl)
                    source = TrainingModel(opt)
                    optimizer = torch.optim.Adam(source.parameters(), lr=0.002)
                    loss = sum(value.square().mean() for value in source.model(features)[0].values())
                    loss = loss + source.loss(torch.ones(2, 4)).square().mean()
                    loss.backward()
                    optimizer.step()
                    checkpoint = str(Path(temp) / ("qattn.pth" if use_bfl else "conv.pth"))
                    save_model(checkpoint, 7, source, optimizer)

                    opt.resume = True
                    restored = TrainingModel(opt)
                    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.1)
                    _, _, epoch = load_model(restored, checkpoint, opt, restored_optimizer)
                    self.assertEqual(epoch, 7)
                    self.assert_states_equal(source.state_dict(), restored.state_dict())
                    expected_state = optimizer.state_dict()
                    actual_state = restored_optimizer.state_dict()
                    self.assertEqual(expected_state["param_groups"], actual_state["param_groups"])
                    self.assertTrue(expected_state["state"])
                    self.assertEqual(set(expected_state["state"]), set(actual_state["state"]))
                    for key in expected_state["state"]:
                        self.assert_states_equal(expected_state["state"][key], actual_state["state"][key])

                    # The production training namespaces must also load into the
                    # inference-only model without bringing the loss classifier.
                    inference_opt = options(use_bfl=use_bfl)
                    inference = FeatureModel(inference_opt)
                    load_model(inference, checkpoint, inference_opt)
                    self.assert_states_equal(source.model.state_dict(), inference.state_dict())

                    other_opt = options(use_bfl=not use_bfl)
                    other_opt.resume = True
                    other = TrainingModel(other_opt)
                    other_optimizer = torch.optim.Adam(other.parameters())
                    with self.assertRaisesRegex(RuntimeError, "(?i)mismatch"):
                        load_model(other, checkpoint, other_opt, other_optimizer)

                    opt.allow_partial_load = True
                    with self.assertRaises((ValueError, RuntimeError)):
                        load_model(restored, checkpoint, opt, restored_optimizer)

    def test_resume_requires_an_explicit_checkpoint(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises((SystemExit, ValueError)):
                opts().parse(["--resume"])
        parsed = opts().parse(["--resume", "--load_model", "/tmp/explicit_checkpoint.pth",
                               "--local_rank", "1"])
        self.assertTrue(parsed.resume)

    def test_public_scripts_select_main_by_default_and_keep_conv_runs_separate(self):
        # Capture the public command lines without starting Python training,
        # reading weights, or requiring a real dataset.
        with tempfile.TemporaryDirectory(prefix="gatemot_decoder_scripts_") as temp:
            temp = Path(temp)
            shim = temp / "python"
            shim.write_text(
                "#!" + sys.executable + "\n"
                "import json, os, sys\n"
                "with open(os.environ['GATEMOT_CAPTURE_ARGS'], 'w') as handle:\n"
                "    json.dump(sys.argv[1:], handle)\n"
            )
            shim.chmod(0o755)
            captured = temp / "args.json"
            env = dict(os.environ)
            for name in ("USE_BFL", "EXP_ID", "ARCH", "ANN_NAME", "SPLIT", "FUSE"):
                env.pop(name, None)
            env.update(PATH=str(temp) + os.pathsep + env.get("PATH", ""),
                       DATA_ROOT=str(temp / "data"), LOAD_MODEL=str(temp / "model.pth"),
                       DEMO=str(temp / "images"), GATEMOT_CAPTURE_ARGS=str(captured),
                       NUM_GPUS="1", SEED="317")
            train_ids = {}
            for script in ("train.sh", "test.sh", "demo.sh"):
                cases = ((None, (), True), ("1", (), True), ("0", (), False),
                         ("0", ("--use_bfl",), True))
                for mode, extra, expected_bfl in cases:
                    with self.subTest(script=script, use_bfl=mode, extra=extra):
                        run_env = dict(env)
                        if mode is not None:
                            run_env["USE_BFL"] = mode
                        command = ["bash", str(ROOT / "scripts" / script)]
                        if script != "demo.sh":
                            command.append("bee24")
                        command.extend(extra)
                        subprocess.run(command, cwd=str(ROOT), env=run_env, check=True,
                                       capture_output=True, text=True, timeout=15)
                        argv = json.loads(captured.read_text())
                        self.assertEqual("--use_bfl" in argv, expected_bfl)
                        if script == "train.sh":
                            experiment = argv[argv.index("--exp_id") + 1]
                            if extra:
                                self.assertEqual(experiment, train_ids["1"])
                            else:
                                train_ids[mode] = experiment
            self.assertEqual(train_ids[None], train_ids["1"])
            self.assertNotEqual(train_ids["1"], train_ids["0"])
            self.assertIn("qattn", train_ids["1"])
            self.assertIn("conv", train_ids["0"])


if __name__ == "__main__":
    unittest.main()
