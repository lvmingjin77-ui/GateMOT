# Software checks

Checked with Linux, Python 3.8.20, PyTorch 2.4.1+cu121, torchvision 0.19.1 and an NVIDIA V100S. CPU and CUDA train/resume/inference tests passed. All four dataset script workflows also passed on CPU with macOS, Python 3.10, PyTorch 2.4.1 and OpenCV 4.10. These cover temporal halves for BEE24/MOT17/MOT20 with DLA-34 and the official train/val directory layout for SportsMOT with DLA-60. Multi-GPU NCCL training has not been tested.

Run from the repository root:

```bash
python tests/test_qattention.py
python tests/test_release.py
python tests/smoke_pipeline.py --dataset bee24 --device cuda
python tests/smoke_pipeline.py --dataset mot17 --device cuda
python tests/smoke_pipeline.py --dataset mot20 --device cuda
python tests/smoke_pipeline.py --dataset sportsmot --device cuda
```

The tests cover Q-Attention gradients and BN fusion, data identities, evaluation frame selection, checkpoint loading, and decoder controls. The smoke test creates small synthetic sequences in a temporary directory, trains for one epoch, resumes for a second, and runs tracking. It checks the selected backbone, annotation paths, saved epoch, and evaluated frames. It requires no downloaded dataset or pretrained checkpoint. Use `--device cpu` for the CPU version. Add `--trackeval /path/to/TrackEval` to run HOTA/CLEAR/Identity through the public evaluation script as well. All four CPU workflows have also passed this TrackEval check.

These are software checks, not benchmark accuracy evaluations.
