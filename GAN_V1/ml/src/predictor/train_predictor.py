from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models

from ml.src.utils.dataset_imagefolder import ImageFolderBinary


def build_model(num_classes=2):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def acc_from_logits(logits, y):
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


def main():
    data_root = Path("ml/data/predictor")
    train_dir = data_root / "train"
    val_dir = data_root / "val"

    out_dir = Path("ml/exports")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "predictor_best.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    img_size = 224
    batch_size = 64  # 4070 Ti can handle this; reduce to 32 if OOM
    epochs = 5       # start small; increase later
    lr = 3e-4
    num_workers = 4

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = ImageFolderBinary(str(train_dir), transform=train_tf)
    val_ds = ImageFolderBinary(str(val_dir), transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    model = build_model(num_classes=2).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_val = -1.0

    print("Device:", device)
    print("Train:", len(train_ds), "Val:", len(val_ds))
    print("Saving best to:", ckpt_path)

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss = tr_acc = 0.0

        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = torch.tensor(y, device=device)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(x)
                loss = crit(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            tr_loss += loss.item()
            tr_acc += acc_from_logits(logits.detach(), y)

        tr_loss /= max(1, len(train_loader))
        tr_acc /= max(1, len(train_loader))

        model.eval()
        va_loss = va_acc = 0.0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(device, non_blocking=True)
                y = torch.tensor(y, device=device)
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    logits = model(x)
                    loss = crit(logits, y)
                va_loss += loss.item()
                va_acc += acc_from_logits(logits, y)

        va_loss /= max(1, len(val_loader))
        va_acc /= max(1, len(val_loader))

        dt = time.time() - t0
        print(f"[{ep:02d}/{epochs}] "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | {dt:.1f}s")

        if va_acc > best_val:
            best_val = va_acc
            torch.save({
                "model_state": model.state_dict(),
                "img_size": img_size,
                "mean": mean,
                "std": std,
                "classes": ["spoof", "real"],
            }, ckpt_path)
            print(f"  ✅ saved best (val acc {best_val:.4f})")

    print("Done.")


if __name__ == "__main__":
    main()
