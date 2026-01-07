from pathlib import Path
import shutil
import zipfile

# ===== CONFIG =====
DATA_ROOT = Path("ml/data")
ZIP_PATH = DATA_ROOT / "archive.zip"

# If you already extracted:
TRAIN_DIR = DATA_ROOT / "train_img"
TEST_DIR  = DATA_ROOT / "test_img"

# Output for your trainer:
OUT_ROOT = DATA_ROOT / "predictor"
USE_SUBFOLDER = "color"   # ignore depth
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

REAL_KEY = "real"         # filename contains "real" => real, else spoof
# ==================


def label_from_filename(name: str) -> str:
    return "real" if REAL_KEY in name.lower() else "spoof"


def copy_image(src: Path, split: str):
    label = label_from_filename(src.name)
    dst_dir = OUT_ROOT / split / label
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if not dst.exists():
        shutil.copy2(src, dst)


def sort_from_folders():
    # Expected extracted layout:
    # ml/data/train_img/color/*.jpg
    # ml/data/test_img/color/*.jpg
    for split_dir, split_name in [(TRAIN_DIR, "train"), (TEST_DIR, "val")]:
        color_dir = split_dir / USE_SUBFOLDER
        if not color_dir.exists():
            print(f"[WARN] Missing folder: {color_dir}")
            continue

        imgs = [p for p in color_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
        print(f"{split_dir.name}/{USE_SUBFOLDER}: {len(imgs)} images")

        for img in imgs:
            copy_image(img, split_name)


def sort_from_zip():
    # Works even if zip paths have extra nesting like:
    # test_img/test_img/color/xxx_real.jpg
    # train_img/color/xxx_fake.jpg
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Zip not found: {ZIP_PATH}")

    print(f"Reading from {ZIP_PATH}")

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        members = zf.namelist()
        count = 0

        for m in members:
            p = Path(m)
            if p.suffix.lower() not in IMG_EXTS:
                continue

            parts = [s.lower() for s in p.parts]

            # must contain train_img or test_img somewhere in the path
            if "train_img" in parts:
                split = "train"
            elif "test_img" in parts:
                split = "val"
            else:
                continue

            # must contain "color" somewhere in the path
            if USE_SUBFOLDER not in parts:
                continue

            filename = p.name
            label = label_from_filename(filename)
            out_dir = OUT_ROOT / split / label
            out_dir.mkdir(parents=True, exist_ok=True)

            out_path = out_dir / filename
            if out_path.exists():
                continue

            with zf.open(m) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

            count += 1

        print(f"Extracted & sorted from zip: {count} images")


def count_outputs():
    def c(p): return len(list(p.glob("*"))) if p.exists() else 0
    print("---- Output counts ----")
    print("train/real :", c(OUT_ROOT / "train" / "real"))
    print("train/spoof:", c(OUT_ROOT / "train" / "spoof"))
    print("val/real   :", c(OUT_ROOT / "val" / "real"))
    print("val/spoof  :", c(OUT_ROOT / "val" / "spoof"))
    print("-----------------------")


def main():
    # Ensure output root exists
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Prefer extracted folders if they exist and have color/
    if (TRAIN_DIR / USE_SUBFOLDER).exists() and (TEST_DIR / USE_SUBFOLDER).exists():
        print("Using extracted folders train_img/ and test_img/ ...")
        sort_from_folders()
    else:
        print("Folders not found (or missing color/). Falling back to zip ...")
        sort_from_zip()

    count_outputs()
    print("✅ Done. Output:", OUT_ROOT.resolve())


if __name__ == "__main__":
    main()
