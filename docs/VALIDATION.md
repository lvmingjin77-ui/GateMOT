# Software checks

The current release passed all four public train → resume → test → TrackEval workflows on CPU with macOS, Python 3.10, PyTorch 2.4.1, torchvision 0.19.1, and OpenCV 4.10. BEE24/MOT17/MOT20 use DLA-34; SportsMOT uses DLA-60. The main protocol uses full train and test, with test GT withheld until training finishes.

Run from the repository root:

```bash
python tests/test_qattention.py
python tests/test_release.py
python tests/smoke_pipeline.py --dataset bee24 --device cpu
python tests/smoke_pipeline.py --dataset mot17 --device cpu
python tests/smoke_pipeline.py --dataset mot20 --device cpu
python tests/smoke_pipeline.py --dataset sportsmot --device cpu
```

The tests cover Q-Attention gradients and BN fusion, task-specific gates, identity labels, temporal split boundaries, evaluation frame selection, and checkpoint loading. Each smoke test uses random initialization and tiny synthetic sequences at 128×128 resolution, trains for one epoch, resumes for a second, and runs tracking. It checks the backbone, annotation paths, saved epoch, and evaluated frames. No dataset or pretrained checkpoint download is needed.

Add `--trackeval /path/to/TrackEval` to include HOTA/CLEAR/Identity evaluation. All four CPU workflows passed this check. Use `--protocol validation` to exercise optional temporal half-splits or SportsMOT's official train/val layout. Use `--device cuda` on an NVIDIA GPU.

Earlier CPU/CUDA train/resume/inference checks passed on Linux with Python 3.8.20, PyTorch 2.4.1+cu121, torchvision 0.19.1, and a V100S. This cleanup preserves both DLA models' parameter keys, shapes, initialized tensors, and forward outputs. Multi-GPU NCCL training has not been tested.

These are software checks, not benchmark accuracy evaluations.
