import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


def build_model(num_classes=2):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--ckpt", default="ml/exports/predictor_best.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location="cpu")

    model = build_model(num_classes=2)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    img_size = ckpt["img_size"]
    mean = ckpt["mean"]
    std = ckpt["std"]

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    x = tf(Image.open(args.img).convert("RGB")).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
    print("spoof:", float(probs[0]), "real:", float(probs[1]))


if __name__ == "__main__":
    main()
