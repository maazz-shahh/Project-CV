from pathlib import Path
import numpy as np
import cv2


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def postprocess_with_opencv(gray_input: np.ndarray, restored_rgb: np.ndarray) -> np.ndarray:
    """
    Unique feature:
    Use edge guidance from grayscale input to boost details in restored output.
    """
    gray_u8 = (gray_input * 255).astype(np.uint8)
    rgb_u8 = np.clip(restored_rgb * 255, 0, 255).astype(np.uint8)

    ycrcb = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_eq = clahe.apply(y)

    edges = cv2.Laplacian(gray_u8, cv2.CV_32F, ksize=3)
    edge_strength = np.clip(np.abs(edges) / (np.max(np.abs(edges)) + 1e-6), 0.0, 1.0)
    edge_strength = cv2.GaussianBlur(edge_strength, (5, 5), 0)

    y_mix = (1.0 - 0.25 * edge_strength) * y + (0.25 * edge_strength) * y_eq
    y_mix = np.clip(y_mix, 0, 255).astype(np.uint8)

    out = cv2.merge([y_mix, cr, cb])
    out = cv2.cvtColor(out, cv2.COLOR_YCrCb2RGB)

    # Keep denoising light to avoid waxy/blurred faces.
    out = cv2.bilateralFilter(out, d=3, sigmaColor=18, sigmaSpace=18)

    # Blend in grayscale luminance for structure; keep gray weight moderate so scratch
    # streaks in damaged inputs do not dominate the final Y channel.
    out_ycrcb = cv2.cvtColor(out, cv2.COLOR_RGB2YCrCb)
    y_out, cr_out, cb_out = cv2.split(out_ycrcb)
    y_detail = np.clip(0.80 * y_out.astype(np.float32) + 0.20 * gray_u8.astype(np.float32), 0, 255).astype(np.uint8)
    out = cv2.cvtColor(cv2.merge([y_detail, cr_out, cb_out]), cv2.COLOR_YCrCb2RGB)

    # Stronger luminance-only unsharp mask (clearer edges, fewer color halos than RGB unsharp).
    ycrcb_sh = cv2.cvtColor(out, cv2.COLOR_RGB2YCrCb)
    y_sh, cr_sh, cb_sh = cv2.split(ycrcb_sh)
    blur_y = cv2.GaussianBlur(y_sh, (0, 0), sigmaX=1.0)
    y_sh = cv2.addWeighted(y_sh, 1.28, blur_y, -0.28, 0)
    y_sh = np.clip(y_sh, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(cv2.merge([y_sh, cr_sh, cb_sh]), cv2.COLOR_YCrCb2RGB)

    return np.clip(out, 0, 255).astype(np.float32) / 255.0


def build_scratch_mask(gray_input: np.ndarray, sensitivity: float = 0.55) -> np.ndarray:
    """
    Estimate thin bright scratch/streak artifacts from grayscale input.
    Returns uint8 mask in [0,255].
    """
    gray_u8 = np.clip(gray_input * 255.0, 0, 255).astype(np.uint8)
    # Top-hat highlights bright thin structures.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    tophat = cv2.morphologyEx(gray_u8, cv2.MORPH_TOPHAT, kernel)
    blur = cv2.GaussianBlur(tophat, (3, 3), 0)
    sens = float(np.clip(sensitivity, 0.15, 0.95))
    # Lower threshold than before to catch brighter scratches aggressively.
    thr = int(255 * (0.35 + 0.45 * sens))
    _, m_tophat = cv2.threshold(blur, thr, 255, cv2.THRESH_BINARY)
    _, m_bright = cv2.threshold(gray_u8, 225, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_or(m_tophat, m_bright)
    # Connect broken lines and remove tiny noise.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask


def remove_scratches_from_rgb(gray_input: np.ndarray, rgb_input: np.ndarray, sensitivity: float = 0.55) -> np.ndarray:
    mask = build_scratch_mask(gray_input, sensitivity=sensitivity)
    rgb_u8 = np.clip(rgb_input * 255.0, 0, 255).astype(np.uint8)
    cleaned = cv2.inpaint(rgb_u8, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return cleaned.astype(np.float32) / 255.0


def clean_gray_input(gray_input: np.ndarray, sensitivity: float = 0.48) -> np.ndarray:
    gray_u8 = np.clip(gray_input * 255.0, 0, 255).astype(np.uint8)
    mask = build_scratch_mask(gray_input, sensitivity=sensitivity)
    cleaned = cv2.inpaint(gray_u8, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return cleaned.astype(np.float32) / 255.0


def boost_color(rgb_input: np.ndarray, sat_gain: float = 1.28, val_gain: float = 1.04) -> np.ndarray:
    rgb_u8 = np.clip(rgb_input * 255.0, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * sat_gain, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * val_gain, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return out.astype(np.float32) / 255.0


def make_comparison_strip(gray, pred, enhanced, gt=None):
    gray3 = np.repeat(gray[..., None], 3, axis=2)
    tiles = [gray3, pred, enhanced]
    if gt is not None:
        tiles.append(gt)
    strip = np.concatenate([np.clip(t, 0, 1) for t in tiles], axis=1)
    return (strip * 255).astype(np.uint8)
