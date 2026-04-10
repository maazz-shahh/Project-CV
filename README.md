# AI-Based Photo Restoration (core pipeline)

Grayscale face → color using a **skip-linked convolutional autoencoder** (TensorFlow) plus **OpenCV** post-processing. This repo keeps the **training + batch evaluation** path only (no web UI).

## Dataset layout

```
dataset/
  gray/    # paired JPGs
  color/   # same filenames as gray/
```

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
py -3.13 train.py --dataset_dir dataset --image_size 128 --batch_size 8 --epochs 10 --max_pairs 3500 --damage_augment
```

Saves best weights to `artifacts/restorer_autoencoder.keras` and a timestamped copy under `artifacts/backups/`.

## Restore & metrics

```bash
py -3.13 restore.py --model artifacts/restorer_autoencoder.keras --input_dir dataset/gray --gt_dir dataset/color --max_images 30 --presentable_mode --save_scale 0.85
```

Writes comparison strips to `outputs/` and prints mean PSNR / SSIM when ground truth exists.

## Possible extensions (not in this tree)

- Streamlit or Gradio frontend  
- Transfer-learning encoder  
- Higher resolution / perceptual loss  
