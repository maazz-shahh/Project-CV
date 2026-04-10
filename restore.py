import argparse
from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm

from src.utils import (
    ensure_dir,
    postprocess_with_opencv,
    make_comparison_strip,
    clean_gray_input,
    remove_scratches_from_rgb,
    boost_color,
)


def read_gray(path: str, image_size: int):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def read_color(path: str, image_size: int):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/restorer_autoencoder.keras")
    parser.add_argument("--input_dir", default="dataset/gray")
    parser.add_argument("--gt_dir", default="dataset/color")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--max_images", type=int, default=30)
    parser.add_argument("--save_scale", type=float, default=0.85, help="Scale factor for saved comparison image size.")
    parser.add_argument("--presentable_mode", action="store_true", help="Apply stronger cleanup and color finishing.")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    model = tf.keras.models.load_model(args.model, compile=False)
    model_image_size = int(model.input_shape[1])
    if model_image_size != args.image_size:
        print(f"Using model input size {model_image_size} (overriding --image_size {args.image_size})")
    run_image_size = model_image_size
    save_scale = float(np.clip(args.save_scale, 0.5, 1.0))

    files = sorted(Path(args.input_dir).glob("*.jpg"))[: args.max_images]
    if not files:
        raise ValueError("No input images found.")

    psnr_vals = []
    ssim_vals = []

    for p in tqdm(files, desc="Restoring"):
        gray = read_gray(str(p), run_image_size)
        if args.presentable_mode:
            gray = clean_gray_input(gray, sensitivity=0.48)
        inp = gray[None, ..., None]
        pred = model.predict(inp, verbose=0)[0]
        enhanced = postprocess_with_opencv(gray, pred)
        if args.presentable_mode:
            enhanced = remove_scratches_from_rgb(gray, enhanced, sensitivity=0.48)
            enhanced = boost_color(enhanced, sat_gain=1.28, val_gain=1.04)

        gt = read_color(str(Path(args.gt_dir) / p.name), run_image_size)
        if gt is not None:
            gt_t = tf.convert_to_tensor(gt[None, ...], tf.float32)
            en_t = tf.convert_to_tensor(enhanced[None, ...], tf.float32)
            psnr_vals.append(float(tf.image.psnr(gt_t, en_t, max_val=1.0)[0].numpy()))
            ssim_vals.append(float(tf.image.ssim(gt_t, en_t, max_val=1.0)[0].numpy()))

        strip = make_comparison_strip(gray, pred, enhanced, gt)
        if save_scale < 1.0:
            h, w = strip.shape[:2]
            strip = cv2.resize(strip, (int(w * save_scale), int(h * save_scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(Path(args.output_dir) / f"{p.stem}_compare.jpg"), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))

    if psnr_vals:
        print(f"Mean PSNR: {np.mean(psnr_vals):.3f}")
        print(f"Mean SSIM: {np.mean(ssim_vals):.3f}")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
