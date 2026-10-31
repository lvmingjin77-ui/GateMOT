# Software checks

The release passed 12 public prepare → train → resume → test → TrackEval workflows on a single NVIDIA V100S: all four datasets with both Q-Attention and Conv, plus BEE24/MOT17 half-validation with both decoders. The environment was Linux, Python 3.8.20, PyTorch 2.4.1+cu121, torchvision 0.19.1, and OpenCV 4.12. BEE24/MOT17/MOT20 use DLA-34; SportsMOT uses DLA-60. Main-protocol checks withheld test GT until tracking finished.

Run from the repository root:

```bash
python tests/test_qattention.py
python tests/test_release.py
python tests/test_dataset_protocols.py
python tests/test_decoder_modes.py
python tests/test_training_numerics.py
python tests/smoke_pipeline.py --dataset bee24 --device cpu
python tests/smoke_pipeline.py --dataset mot17 --device cpu
python tests/smoke_pipeline.py --dataset mot20 --device cpu
python tests/smoke_pipeline.py --dataset sportsmot --device cpu
```

The tests cover Q-Attention and Conv selection, gradients and BN fusion, strict checkpoint loading and resume, task-specific gates, split-local identity labels, MOT17 FRCNN filtering, odd-length temporal split boundaries, evaluation frame selection, SIoU at zero center distance, and rotated previous-frame boxes. Each smoke test uses random initialization and tiny synthetic sequences at 128×128 resolution, trains for one epoch, resumes for a second, and runs tracking. It checks the decoder, backbone, annotation paths, saved epoch, and evaluated frames. No dataset or pretrained checkpoint download is needed. Add `--decoder conv` to run the same workflow with the Conv baseline.

Add `--trackeval /path/to/TrackEval` to include HOTA/CLEAR/Identity evaluation. Use `--protocol validation` to exercise optional temporal half-splits or SportsMOT's official train/val layout. Use `--device cuda` on an NVIDIA GPU.

Unit tests also passed on macOS CPU with Python 3.10, PyTorch 2.4.1, torchvision 0.19.1, and OpenCV 4.10. The restored decoder heads preserve the original parameter keys, shapes, initialized tensors, and forward outputs for both DLA models. Multi-GPU NCCL training has not been tested.

These are software checks, not benchmark accuracy evaluations.
