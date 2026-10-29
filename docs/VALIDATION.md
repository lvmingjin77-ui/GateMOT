# Software checks

Checked with Linux, Python 3.8.20, PyTorch 2.4.1+cu121, torchvision 0.19.1 and an NVIDIA V100S. CPU and CUDA train/resume/inference tests passed. Multi-GPU NCCL training has not been tested.

Run from the repository root:

```bash
python tests/test_qattention.py
python tests/test_release.py
python tests/smoke_pipeline.py --device cuda
```

The tests cover Q-Attention gradients and BN fusion, data identities, evaluation frame selection, checkpoint loading, and decoder controls. The smoke test creates small synthetic sequences in a temporary directory, trains for one epoch, resumes for a second, and runs tracking. It requires no downloaded dataset or pretrained checkpoint. Use `--device cpu` for the CPU version.

These are software checks, not benchmark accuracy evaluations.
