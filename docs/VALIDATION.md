# Software checks

Checked with Linux, Python 3.8.20, PyTorch 2.4.1+cu121, torchvision 0.19.1 and an NVIDIA V100S. CPU and CUDA train/resume/inference tests passed. The three dataset script workflows also passed on macOS with Python 3.10, PyTorch 2.4.1 and OpenCV 4.10. Multi-GPU NCCL training has not been tested.

Run from the repository root:

```bash
python tests/test_qattention.py
python tests/test_release.py
python tests/smoke_pipeline.py --dataset bee24 --device cuda
python tests/smoke_pipeline.py --dataset mot17 --device cuda
python tests/smoke_pipeline.py --dataset mot20 --device cuda
```

The tests cover Q-Attention gradients and BN fusion, data identities, evaluation frame selection, checkpoint loading, and decoder controls. The smoke test creates small synthetic sequences in a temporary directory, trains for one epoch, resumes for a second, and runs tracking. It requires no downloaded dataset or pretrained checkpoint. Use `--device cpu` for the CPU version. Add `--trackeval /path/to/TrackEval` to run HOTA/CLEAR/Identity through the public evaluation script as well.

These are software checks, not benchmark accuracy evaluations.
