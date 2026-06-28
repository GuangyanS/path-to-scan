# path-to-scan

Minimal PyTorch training code for a 4-channel ALL-CNN-C model on scan-generated
CIFAR-100 RAW/RGGB H5 data.

The default model is the version that worked best in our controlled run:
`Conv -> BatchNorm2d -> ReLU` for the first 8 convolution layers, then a final
logits conv plus global average pooling.

## Data

Put the generated H5 file here:

```bash
datasets/cifar100_raw.h5
```

The loader expects:

- `images`: `(N, 4, H, W)` or `(N, H, W, 4)`
- `labels`: `(N,)`
- `train`: boolean split mask

For CIFAR-100 this should be 50,000 train images and 10,000 test images.

## Environment

On the shared GPU machine, use the system CUDA PyTorch install:

```bash
uv venv --python /usr/bin/python3.10 --system-site-packages .venv
uv run --no-sync python -m py_compile config.py h5_dataset.py main.py models/ALL_CNN_C.py
```

## Train

Use GPU 4 only:

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 uv run python main.py train
```

The defaults in `config.py` are set to the tested BN recipe:

- 4-channel input, 100 classes
- `Conv -> BatchNorm2d -> ReLU` in the first 8 conv layers
- H5 data source: `./datasets/cifar100_raw.h5`
- RAW augmentation and D300 pointwise noise enabled
- SGD, batch size 128, lr 0.1, weight decay 1e-4
- 5 epoch warmup, multistep milestones `100,150`
- 200 epochs, best checkpoint saved to `checkpoints/`

## Test

```bash
CUDA_VISIBLE_DEVICES=4 uv run python main.py test \
  --checkpoint_load_name='ALL_CNN_C_c100_rggb_h5_bn_refstyle'
```

For clean test data, disable test-time noise:

```bash
CUDA_VISIBLE_DEVICES=4 uv run python main.py test \
  --checkpoint_load_name='ALL_CNN_C_c100_rggb_h5_bn_refstyle' \
  --raw_noise=False
```

## Full-Precision PCN

`ALL_CNN_C_PCN` adds local predictive-coding recurrences to the first eight
ALL-CNN-C conv layers. The recurrent update uses learned non-negative per-filter
`alpha`, feedback transposed convolutions, and 1x1 bypass convolutions.

Start from the FP32 ALL-CNN-C checkpoint with non-strict loading because the PCN
feedback, bypass, and alpha tensors are new:

```bash
CUDA_VISIBLE_DEVICES=4 uv run --no-sync python main.py test \
  --model=ALL_CNN_C_PCN \
  --pcn_cycles=3 \
  --checkpoint_load_name=ALL_CNN_C_c100_rggb_h5_bn_refstyle \
  --checkpoint_load_strict=False \
  --raw_noise=False \
  --num_workers=0
```

Fine-tune from the same checkpoint:

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 uv run --no-sync python main.py train \
  --model=ALL_CNN_C_PCN \
  --pcn_cycles=3 \
  --use_trained_model=True \
  --checkpoint_load_name=ALL_CNN_C_c100_rggb_h5_bn_refstyle \
  --checkpoint_load_strict=False \
  --checkpoint_save_name=ALL_CNN_C_PCN_c100_rggb_h5_t3_ft \
  --lr=0.001 \
  --warmup=0
```

Initial GPU 4 comparison with `num_workers=0`:

- baseline clean: `61.18%`
- PCN checkpoint-start clean: `61.15%`
- PCN after 1 epoch fine-tune clean: `61.20%`
- baseline noisy: `61.31%`
- PCN after 1 epoch fine-tune noisy: `61.51%`

## 4W4A QAT

Fine-tune the 4-bit model from the FP32 checkpoint:

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 uv run --no-sync python main.py qat_finetune
```

QAT uses fake quantization with STE during training, then `real_quant_test`
materializes integer codes for evaluation:

- folded Conv+BN weights use signed int4 codes in `[-7, 7]`
- activations use unsigned int4 codes in `[0, 15]`
- Conv bias after BN folding remains FP32

Evaluate the saved QAT checkpoint:

```bash
CUDA_VISIBLE_DEVICES=4 uv run --no-sync python main.py real_quant_test --raw_noise=False
CUDA_VISIBLE_DEVICES=4 uv run --no-sync python main.py real_quant_test
```

For integer-domain simulation, use:

```bash
uv run --no-sync python main.py int4_sim_test --raw_noise=False --use_gpu=False --num_workers=0
```

This path lives in `simulator/`. It keeps activations as uint4 codes, weights
as signed int4 codes, uses int32 conv accumulation, and requantizes with
precomputed per-layer constants. It runs on CPU because PyTorch does not provide
a CUDA int4/int32 conv kernel.

## QKD

The QKD code is self-contained. Put the EfficientNetV2-L teacher checkpoint at:

```bash
checkpoints/b4_100.pth
```

Then verify the teacher:

```bash
CUDA_VISIBLE_DEVICES=4 uv run --no-sync python main.py teacher_test --raw_noise=False --num_workers=0 --batch_size=16
```

Run stages with `qkd_finetune` and set `qkd_stage` to `SS`, `CS`, or `TU`.

```bash
CUDA_VISIBLE_DEVICES=4 uv run --no-sync python main.py qkd_finetune --qkd_stage=SS
CUDA_VISIBLE_DEVICES=4 uv run --no-sync python main.py qkd_finetune --qkd_stage=CS
CUDA_VISIBLE_DEVICES=4 uv run --no-sync python main.py qkd_finetune --qkd_stage=TU
```

By default, SS saves `*_qkd_ss.pth`, CS loads that and saves `*_qkd_cs.pth`,
and TU loads `*_qkd_cs.pth`.

For the current best minimal run, TU was started from the SS student while using
the co-studied teacher:

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 uv run --no-sync python main.py qkd_finetune \
  --qkd_stage=TU \
  --qkd_student_checkpoint_name=ALL_CNN_C_c100_rggb_h5_w4a4_qkd_ss
```

## Architecture

Input: `4 x 16 x 16`

| Layer | Operation | Cin -> Cout | Kernel | Stride | Output size |
| --- | --- | ---: | ---: | ---: | ---: |
| 1.1 | Conv+BN+ReLU | 4 -> 96 | 3x3 | 1 | 96 x 16 x 16 |
| 1.2 | Conv+BN+ReLU | 96 -> 96 | 3x3 | 1 | 96 x 16 x 16 |
| 1.3 | Conv+BN+ReLU | 96 -> 96 | 3x3 | 2 | 96 x 8 x 8 |
| 2.1 | Conv+BN+ReLU | 96 -> 192 | 3x3 | 1 | 192 x 8 x 8 |
| 2.2 | Conv+BN+ReLU | 192 -> 192 | 3x3 | 1 | 192 x 8 x 8 |
| 2.3 | Conv+BN+ReLU | 192 -> 192 | 3x3 | 2 | 192 x 4 x 4 |
| 3 | Conv+BN+ReLU | 192 -> 192 | 3x3 | 1 | 192 x 4 x 4 |
| 4 | Conv+BN+ReLU | 192 -> 192 | 1x1 | 1 | 192 x 4 x 4 |
| 5 | Conv logits + GAP | 192 -> 100 | 1x1 | 1 | 100 x 4 x 4 -> 100 |

## Current Reference Result

On `cifar100_raw.h5`, using GPU 4 and the default BN recipe:

- training-log best test accuracy: `61.44%`
- best checkpoint epoch: `173`
- fixed clean test: `61.18%`
- noisy test sanity check: `61.31%` with `--num_workers=0`

For 4W4A QAT from the FP32 checkpoint:

- initial fake-quant clean test: `34.65%`
- QAT training-log best noisy test: `56.84%`
- best QAT checkpoint epoch: `34`
- real quant clean test: `57.09%`
- real quant noisy test: `56.60%`

For the self-contained QKD flow:

- `b4_100.pth` teacher is 4-channel RAW/RGGB, not RGB
- original teacher eval: `77.30%` clean, `77.33%` noisy
- one-epoch co-studied teacher eval: `78.70%` noisy
- SS best noisy test: `56.11%`
- TU best checkpoint history: `56.27%`
- QAT eval with `num_workers=0`: `56.02%` noisy, `56.17%` clean
- true INT4 simulator: `56.02%` noisy, `56.17%` clean
- simulator code ranges: weights `[-7, 7]`, activations `[0, 15]`
