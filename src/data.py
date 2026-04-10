from pathlib import Path
import random
import tensorflow as tf


AUTOTUNE = tf.data.AUTOTUNE


def _read_image(path: tf.Tensor, channels: int, image_size: int) -> tf.Tensor:
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=channels)
    img = tf.image.resize(img, (image_size, image_size), method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _load_pair(gray_path: tf.Tensor, color_path: tf.Tensor, image_size: int):
    gray = _read_image(gray_path, channels=1, image_size=image_size)
    color = _read_image(color_path, channels=3, image_size=image_size)
    return gray, color


def _random_damage(gray: tf.Tensor) -> tf.Tensor:
    # Random brightness/contrast jitter
    x = tf.image.random_brightness(gray, max_delta=0.08)
    x = tf.image.random_contrast(x, lower=0.8, upper=1.2)

    # Add Gaussian noise
    noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=0.04)
    x = x + noise

    # Vertical banding artifact simulation
    w = tf.shape(x)[1]
    stripe = tf.random.normal([1, w, 1], mean=0.0, stddev=0.06)
    stripe = tf.repeat(stripe, repeats=tf.shape(x)[0], axis=0)
    stripe = tf.expand_dims(stripe, axis=-1)
    stripe = tf.squeeze(stripe, axis=-1)
    x = x + stripe

    # Sparse bright scratch-like artifacts
    scratch = tf.cast(tf.random.uniform(tf.shape(x)) > 0.992, tf.float32)
    scratch = tf.nn.max_pool2d(scratch[None, ...], ksize=3, strides=1, padding="SAME")[0]
    x = tf.where(scratch > 0.0, tf.maximum(x, 0.9), x)

    return tf.clip_by_value(x, 0.0, 1.0)


def list_pairs(dataset_dir: str):
    base = Path(dataset_dir)
    gray_dir = base / "gray"
    color_dir = base / "color"
    if not gray_dir.exists() or not color_dir.exists():
        raise FileNotFoundError("Expected dataset/{gray,color} directories.")

    gray_files = {p.name: str(p) for p in gray_dir.glob("*.jpg")}
    color_files = {p.name: str(p) for p in color_dir.glob("*.jpg")}
    common = sorted(set(gray_files.keys()) & set(color_files.keys()))
    if not common:
        raise ValueError("No paired images found by matching filenames.")
    return [gray_files[n] for n in common], [color_files[n] for n in common]


def build_datasets(
    dataset_dir: str,
    image_size: int = 128,
    batch_size: int = 16,
    val_split: float = 0.1,
    seed: int = 42,
    max_pairs: int | None = None,
    damage_augment: bool = True,
):
    gray_paths, color_paths = list_pairs(dataset_dir)
    if max_pairs is not None and max_pairs > 0:
        gray_paths = gray_paths[:max_pairs]
        color_paths = color_paths[:max_pairs]
    idx = list(range(len(gray_paths)))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    gray_paths = [gray_paths[i] for i in idx]
    color_paths = [color_paths[i] for i in idx]

    n_val = max(1, int(len(gray_paths) * val_split))
    train_gray, val_gray = gray_paths[n_val:], gray_paths[:n_val]
    train_color, val_color = color_paths[n_val:], color_paths[:n_val]

    def _mk(gp, cp, training=True):
        ds = tf.data.Dataset.from_tensor_slices((gp, cp))
        if training:
            ds = ds.shuffle(min(len(gp), 4000), seed=seed)
        ds = ds.map(
            lambda g, c: _load_pair(g, c, image_size=image_size),
            num_parallel_calls=AUTOTUNE,
        )
        if training and damage_augment:
            ds = ds.map(lambda g, c: (_random_damage(g), c), num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(AUTOTUNE)
        return ds

    return _mk(train_gray, train_color, True), _mk(val_gray, val_color, False)
