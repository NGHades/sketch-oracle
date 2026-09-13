"""Download per-category 28x28 bitmap arrays from Google's Quick, Draw! public bucket."""
import argparse
from pathlib import Path
from urllib.parse import quote

import requests
from tqdm import tqdm

BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy"


def download_category(category: str, out_dir: Path) -> None:
    dest = out_dir / f"{category}.npy"
    if dest.exists():
        return
    url = BASE_URL.format(quote(category))
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=category, leave=False
    ) as bar:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
            bar.update(len(chunk))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", default="categories.txt", help="File with one category name per line")
    parser.add_argument("--out-dir", default="data/raw", help="Where to store downloaded .npy files")
    args = parser.parse_args()

    categories = [line.strip() for line in Path(args.categories).read_text().splitlines() if line.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for category in tqdm(categories, desc="categories"):
        try:
            download_category(category, out_dir)
        except requests.HTTPError as e:
            print(f"skipped '{category}': {e}")

    print(f"done. {len(list(out_dir.glob('*.npy')))} files in {out_dir}")


if __name__ == "__main__":
    main()
