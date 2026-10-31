# Implementation

Implementation details for the DLA models and the defaults in `scripts/train.sh` and `scripts/test.sh`.

## Q-Attention

`src/lib/model/networks/base_model.py` implements each task-specific operator. With input `X`, the checkpoint names `conv1`, `conv2`, and `conv3` are the Value, Key, and Query projections respectively.

`V = Wv(X)`, `K = Wk(X)`, `G = sigmoid(Wq(X))`

`Y = BN(Wf(concat(V, MaxPool3x3(K * G))))`

All projections/fusion are 1×1 convolutions. The pool has stride 1 and padding 1. There is no global affinity matrix or spatial softmax. The ungated Value is retained in both normal and BN-fused inference. A ReLU and final 1×1 predictor follow the transform. Query selection and local aggregation remain separate from global pairwise aggregation, which is absent.

DLA-34 supplies 64 input channels and DLA-60 supplies 128. Both use 256 intermediate channels. `--num_head_conv 1` is the provided configuration. Larger values add 1×1 transforms after the first Q-Attention block, not a stack of independent Q-Attention blocks.

The original `--use_bfl` flag selects Q-Attention. Without it, each head uses a 3×3 Conv → ReLU → 1×1 predictor, with no BatchNorm in the head. `--head_kernel` changes only the first Conv baseline layer. The public train/test/demo scripts enable `--use_bfl` by default; set `USE_BFL=0` for the Conv baseline. Use the same decoder setting for training, resume, and inference. All other task heads, losses, and association settings are shared.

| Head | Output | Role |
|---|---:|---|
| hm | number of classes | center heatmap |
| reg | 2 | sub-cell center offset |
| ltrb_amodal | 4 | box sides relative to center |
| tracking | 2 | previous-frame displacement |
| reid | 128 | appearance embedding |
| wh | 2 | auxiliary width/height training output |

Inference decodes boxes from `ltrb_amodal`, not the auxiliary `wh` head. Default online association requires the detection, motion, and ReID outputs.

## Encoder and DCN

Current RGB, previous RGB, and previous center heatmap pass through separate stems and are added.

Modulated DCN uses `torchvision.ops.deform_conv2d`, retaining the original parameter names, offset/mask projection, and tensor shapes. No separately compiled DCNv2 extension is needed.

## Loss and optimization

With the supplied scripts, the loss before the final batch multiplier is

`L = Lhm + Lreg + Ltracking + 0.1 * Lltrb_L1 + 0.1 * Lwh + wReID * LReID + 2 * LSIoU`.

`wReID` is 1.0 for all four datasets. The total loss is multiplied by the **per-process batch size**. The trainable identity classifier belongs to the loss module and is not an inference parameter. Identity count is derived from the selected training JSON.

Adam defaults to zero weight decay. LR is multiplied by 0.1 after each listed decay epoch. `--warmup_epochs N` optionally applies epoch-wise linear warmup. Defaults do not enable it. Seeds cover Python, NumPy and PyTorch. Distributed samplers are reseeded each epoch. Resume restores optimizer state but not random-generator states.

## Augmentation

Random crop is enabled in the scripts, drawing scale from 0.6, 0.7, …, 1.3. Shift/scale affine jitter is the alternative when `--not_rand_crop` is supplied. BEE24 rotation is Gaussian with standard deviation 15°, clipped to ±30°, applied with probability 1. SportsMOT disables horizontal flips, while other scripts use probability 0.5. Color augmentation uses CornerNet brightness, contrast and saturation scale 0.4 plus lighting scale 0.1. The image transform uses a centered affine warp. Previous-frame heatmap disturbance settings are 0.05 positional noise, 0.4 dropped centers, and 0.1 false centers.

## Association

Online association is implemented in `src/lib/tracker_zoo/dctrack.py`.

| Operation | Implementation default |
|---|---|
| Detection threshold | 0.4 |
| New track threshold | 0.4 |
| Motion assignment | IoU cost ≤0.75, equivalent to IoU ≥0.25 |
| Lost-track appearance assignment | cosine distance ≤0.35, equivalent to similarity ≥0.65 |
| Appearance geometric gate | IoU cost ≤0.9 |
| Unconfirmed assignment | cost ≤0.7 |
| Lost-track lifetime | 20 frames BEE24, 30 otherwise |
| Box/center EMA | new estimate weight 0.35, rising to 0.65 after 5 updates |
| Unconfirmed retention threshold | score ≥0.55 |

The buffer implementation can move an unmatched high-score unconfirmed track to the lost pool, where `max_age` applies.

The appearance update is `normalize(0.4 * (1-score) * old + 0.6 * score * new)`. Motion targets are `previous_center - current_integer_cell`; postprocessing accounts for the refined current center, and back-projection adds the resulting offset.

## Evaluation

The annotation manifest is the source of truth for frame membership. Frames with no predicted objects still count as misses when GT objects exist. Internal CLEAR evaluation is diagnostic. TrackEval bundles consistently filter and renumber GT and predictions for half-validation, preserving the originals. MOT17/MOT20 use MOTChallenge preprocessing. Other supported datasets are treated as a single tracked class without MOT pedestrian-class preprocessing.

## Version notes

The current scripts and the submitted supplementary description differ in several settings. The implementation above uses separate input stems, L1 plus SIoU box supervision, zero default warmup/weight decay, and the association thresholds listed here. Its appearance update and rotation sampler are also specified above. These settings should be used when running this release. For downloaded weights, use the matching inference settings and validation notes in the [checkpoint guide](MODELS.md).

Annotations generated by earlier converters should be regenerated. Identity labels are now unique per sequence, and evaluation includes frames with no predictions. These changes can affect training and evaluation results from earlier code versions.
