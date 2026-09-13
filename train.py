"""Train the sketch classifier on the prepared train/val .npz splits."""
import argparse
from pathlib import Path

import numpy as np
from tensorflow import keras

from model import build_model


def load_split(processed_dir: Path, name: str):
    data = np.load(processed_dir / f"{name}.npz")
    return data["X"], data["y"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out", default="models/sketch_oracle.keras")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    classes = [line.strip() for line in (data_dir / "classes.txt").read_text().splitlines() if line.strip()]

    X_train, y_train = load_split(data_dir, "train")
    X_val, y_val = load_split(data_dir, "val")

    model = build_model(num_classes=len(classes))
    model.summary()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=3, min_lr=1e-5),
        keras.callbacks.ModelCheckpoint(str(out_path), monitor="val_accuracy", save_best_only=True),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
    )

    print(f"best model saved to {out_path}")


if __name__ == "__main__":
    main()
