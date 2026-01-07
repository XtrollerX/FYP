import torch
import torch.nn as nn
from torchvision import models

def build_predictor(num_classes=2):
    # must match your predictor architecture (resnet18 in our baseline)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def load_predictor(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    classes = ckpt.get("classes", ["spoof", "real"])
    img_size = ckpt.get("img_size", 224)
    mean = ckpt.get("mean", (0.485, 0.456, 0.406))
    std = ckpt.get("std", (0.229, 0.224, 0.225))

    model = build_predictor(num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    return model, classes, img_size, mean, std
