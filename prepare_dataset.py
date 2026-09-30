"""Balance, split, and pack the downloaded per-category .npy bitmaps into train/val/test .npz files."""
import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw", help="Directory of per-category .npy files")
    parser.add_argument("--out-dir", default="data/processed", help="Where to write train/val/test .npz files")
    parser.add_argument("--per-class", type=int, default=12000, help="Samples to keep per category")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    files = sorted(raw_dir.glob("*.npy"))
    if not files:
        raise SystemExit(f"no .npy files found in {raw_dir} — run download_data.py first")

    classes = [f.stem for f in files]
    rng = np.random.default_rng(args.seed)

    images, labels = [], []
    for label_idx, f in enumerate(files):
        arr = np.load(f)  # (N, 784) uint8, values 0-255
        if len(arr) > args.per_class:
            idx = rng.choice(len(arr), args.per_class, replace=False)
            arr = arr[idx]
        images.append(arr)
        labels.append(np.full(len(arr), label_idx, dtype=np.int32))

    # kept as uint8 (not normalized) to keep the in-memory/on-disk footprint 4x smaller;
    # normalization happens inside the model's Rescaling layer instead.
    X = np.concatenate(images).reshape(-1, 28, 28, 1).astype("uint8", copy=False)
    y = np.concatenate(labels)
    del images, labels  # free the per-class copies before splitting duplicates X again

    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=args.val_frac + args.test_frac, stratify=y, random_state=args.seed
    )
    rel_test = args.test_frac / (args.val_frac + args.test_frac)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=rel_test, stratify=y_rest, random_state=args.seed
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "train.npz", X=X_train, y=y_train)
    np.savez_compressed(out_dir / "val.npz", X=X_val, y=y_val)
    np.savez_compressed(out_dir / "test.npz", X=X_test, y=y_test)
    (out_dir / "classes.txt").write_text("\n".join(classes) + "\n")

    print(f"classes: {len(classes)}")
    print(f"train: {X_train.shape}  val: {X_val.shape}  test: {X_test.shape}")


if __name__ == "__main__":
    main()
