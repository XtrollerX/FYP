from pathlib import Path
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageFolderBinary(Dataset):
    """
    Expects:
      root/
        real/  (label 1)
        spoof/ (label 0)

    Returns (image, label_int, path_str)
    """
    def __init__(self, root_dir: str, transform=None):
        self.root = Path(root_dir)
        self.transform = transform

        self.samples: List[Tuple[str, int]] = []

        # label convention: spoof=0, real=1
        for cls_name, label in [("spoof", 0), ("real", 1)]:
            cls_dir = self.root / cls_name
            if not cls_dir.exists():
                continue
            for p in cls_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    self.samples.append((str(p), label))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found in {root_dir}. Expected folders real/ and spoof/ with images."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, path
