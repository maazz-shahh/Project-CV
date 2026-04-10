import argparse
import shutil
from datetime import datetime
from pathlib import Path
import tensorflow as tf

from src.data import build_datasets
from src.model import baseline_autoencoder, combined_loss


def main():
    parser = argparse.ArgumentParser(description="Train skip-connection gray→RGB autoencoder.")
    parser.add_argument("--dataset_dir", default="dataset")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--out_dir", default="artifacts")
    parser.add_argument("--max_pairs", type=int, default=0, help="Use only first N pairs (0 = all).")
    parser.add_argument("--damage_augment", action="store_true", help="Synthetic noise/scratches on inputs.")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    model_path = Path(args.out_dir) / "restorer_autoencoder.keras"

    train_ds, val_ds = build_datasets(
        dataset_dir=args.dataset_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_split=0.1,
        max_pairs=(args.max_pairs if args.max_pairs > 0 else None),
        damage_augment=args.damage_augment,
    )

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            str(model_path), save_best_only=True, monitor="val_loss", verbose=1
        ),
    ]

    model = baseline_autoencoder(input_shape=(args.image_size, args.image_size, 1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=combined_loss,
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    print("Architecture: skip-linked baseline autoencoder")
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    if model_path.is_file():
        bak_dir = Path(args.out_dir) / "backups"
        bak_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = bak_dir / f"{model_path.stem}_{stamp}.keras"
        shutil.copy2(model_path, bak)
        print(f"Backup copy: {bak}")

    print(f"Saved best model to: {model_path}")


if __name__ == "__main__":
    main()
