"""Regression tests for data identity, evaluation frames and decoder controls."""
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'src/lib'),str(ROOT/'src/tools')]
from mot_format import convert
from prepare_trackeval import prepare
from opts import opts
from model.model import create_model, load_model
from model.networks.base_model import QAttention
from model.utils import fuse_model
from tracking_utils.evaluation import Evaluator
from dataset.datasets.custom_dataset import CustomDataset


def fixture(root):
    for split in ('train','val','test'):
        for name in ('SEQ_A','SEQ_B'):
            seq=root/split/name
            (seq/'img1').mkdir(parents=True)
            (seq/'gt').mkdir()
            rows=[]
            for frame in range(1,5):
                cv2.imwrite(str(seq/'img1'/f'{frame:08d}.jpg'),np.full((64,64,3),120,dtype=np.uint8))
                for tid in (1,2):
                    rows.append(f'{frame},{tid},{tid*15},15,10,10,1,1,1\n')
            (seq/'gt/gt.txt').write_text(''.join(rows))


def options(extra=()):
    parser=opts()
    return parser.update_res_info_and_set_heads(parser.parse([
        '--use_bfl','--wh','--pre_hm','--no_pretrained',
        '--input_h','128','--input_w','128']+list(extra)))


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='gatemot_test_')
        self.root=Path(self.tmp.name)
        fixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_identity_is_stable_and_video_specific(self):
        for dataset in ('bee','mot17','mot20','sportsmot','dancetrack'):
            convert(dataset,self.root,['train'])
            data=json.loads((self.root/'annotations/train.json').read_text())
            video={x['id']:x['video_id'] for x in data['images']}
            pairs={}
            for ann in data['annotations']:
                key=(video[ann['image_id']],ann['source_track_id'])
                pairs.setdefault(key,set()).add(ann['track_id'])
            self.assertTrue(all(len(v)==1 for v in pairs.values()))
            self.assertEqual(len({next(iter(v)) for v in pairs.values()}),4)
        opt=options(['--custom_dataset_img_path',str(self.root/'train'),
                     '--custom_dataset_ann_path',str(self.root/'annotations/train.json')])
        # No access to GT is needed after conversion, including the unused split.
        for path in self.root.glob('*/SEQ_*/gt/gt.txt'):
            path.unlink()
        dataset=CustomDataset(opt,'train')
        self.assertEqual(opt.nid,4)
        self.assertEqual(len(dataset),8)

    def test_half_split_and_trackeval_bundle(self):
        convert('bee',self.root)
        train=json.loads((self.root/'annotations/halftrain.json').read_text())
        val=json.loads((self.root/'annotations/halfval.json').read_text())
        self.assertFalse({x['file_name'] for x in train['images']} & {x['file_name'] for x in val['images']})
        tracks=self.root/'tracks'
        tracks.mkdir()
        for name in ('SEQ_A','SEQ_B'):
            (tracks/(name+'.txt')).write_text((self.root/'train'/name/'gt/gt.txt').read_text())
        prepare(self.root/'train',self.root/'annotations/halfval.json',tracks,self.root/'bundle','bee24')
        lines=(self.root/'bundle/gt/SEQ_A/gt/gt.txt').read_text().splitlines()
        self.assertEqual(len(lines),4)
        self.assertEqual({line.split(',')[0] for line in lines},{'1','2'})

    def test_empty_prediction_frames_count_as_misses(self):
        prediction=self.root/'pred.txt'
        prediction.write_text('1,1,15,15,10,10,1,-1,-1,-1\n')
        evaluator=Evaluator(str(self.root/'train'),'SEQ_A','mot')
        import motmetrics as mm
        summary=mm.metrics.create().compute(evaluator.eval_file(str(prediction)),metrics=['num_frames','num_misses'])
        self.assertEqual(int(summary.iloc[0]['num_frames']),4)
        self.assertEqual(int(summary.iloc[0]['num_misses']),7)
        summary=mm.metrics.create().compute(evaluator.eval_file(str(prediction),frame_ids=[3,4]),metrics=['num_frames','num_misses'])
        self.assertEqual(int(summary.iloc[0]['num_frames']),2)
        self.assertEqual(int(summary.iloc[0]['num_misses']),4)

    def test_qattention_backward_and_bn_fusion(self):
        layer=QAttention(64,256).eval()
        x=torch.randn(2,64,8,8,requires_grad=True)
        expected=layer(x)
        torch.testing.assert_close(expected,fuse_model(copy.deepcopy(layer))(x),rtol=1e-4,atol=1e-5)
        expected.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_head_override_retains_outputs_and_losses(self):
        opt=options(['--head_overrides','tracking:conv_bn'])
        model=create_model(opt.arch,opt.heads,opt.head_conv,opt)
        self.assertEqual(set(model.heads),{'hm','reg','tracking','ltrb_amodal','wh','reid'})
        self.assertIsInstance(model.hm[0],QAttention)
        self.assertIsInstance(model.tracking[0][1],torch.nn.BatchNorm2d)
        self.assertGreater(opt.weights['tracking'],0)

    def test_affine_augmentation_accepts_scalar_and_vector_scale(self):
        from dataset.generic_dataset import GenericDataset
        dataset=GenericDataset()
        dataset.opt=options(['--not_rand_crop'])
        for scale in (64.0,np.array([64.,48.],dtype=np.float32)):
            center,aug,rotation=dataset._get_aug_param(np.array([32.,32.]),scale,64,64)
            self.assertTrue(np.isfinite(center).all())
            self.assertTrue(np.isfinite(aug))

    def test_gating_controls_propagate_finite_gradients(self):
        from model.networks.gating_controls import GatingControl
        for size in (1,3):
            layer=GatingControl(8,16,size)
            x=torch.randn(2,8,8,8,requires_grad=True)
            out=layer(x)
            self.assertEqual(tuple(out.shape),(2,16,8,8))
            out.square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_checkpoint_mismatch_is_rejected(self):
        opt=options()
        model=torch.nn.Linear(2,2)
        path=self.root/'wrong.pth'
        torch.save({'epoch':1,'state_dict':{'weight':torch.zeros(3,3)}},path)
        with self.assertRaises(RuntimeError):
            load_model(model,str(path),opt)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
