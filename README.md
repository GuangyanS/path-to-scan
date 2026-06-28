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
