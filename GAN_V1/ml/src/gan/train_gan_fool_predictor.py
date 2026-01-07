import os
from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from ml.src.utils.dataset_imagefolder import ImageFolderBinary
from ml.src.gan.models_gan import Generator, Discriminator
from ml.src.gan.load_predictor import load_predictor


def to_predictor_input(x_tanh, mean, std):
    """
    x_tanh: [-1,1] (GAN output)
    convert -> [0,1] then normalize with ImageNet mean/std for predictor
    """
    x01 = (x_tanh + 1) / 2.0
    mean = torch.tensor(mean, device=x01.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=x01.device).view(1, 3, 1, 1)
    return (x01 - mean) / std


def main():
    # ----- paths -----
    data_root = Path("ml/data/predictor/train")
    ckpt_path = "ml/exports/predictor_best.pt"
    out_dir = Path("ml/exports/gan")
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = out_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # ----- config -----
    z_dim = 128
    batch_size = 16
    lr_g = 2e-4
    lr_d = 2e-4
    epochs = 10
    num_workers = 4

    # weight for fooling predictor
    lam_fool = 0.2   #start small increase later (0.2 to start increase to 1 later) this is to make sure G dosnt ignore realism

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    # ----- load predictor (frozen) -----
    predictor, classes, img_size, mean, std = load_predictor(ckpt_path, device=device)
    for p in predictor.parameters():
        p.requires_grad = False

    real_index = classes.index("real") if "real" in classes else 1  # target class

    print("Device:", device)
    print("Predictor classes:", classes, "| img_size:", img_size)

    # ----- GAN dataset: use ONLY real images as "real" distribution -----
    # We'll point dataset root at train/, and only take real folder
    # dataset_imagefolder expects root/real + root/spoof, so easiest is:
    # use whole dataset but filter by label inside loop.
    tf_gan = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -> [-1,1]
    ])

    ds = ImageFolderBinary(str(data_root), transform=tf_gan)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    # ----- models -----
    G = Generator(z_dim=z_dim, img_channels=3, feature_g=64, img_size=img_size).to(device)
    D = Discriminator(img_channels=3, feature_d=64).to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))

    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    fixed_z = torch.randn(32, z_dim, 1, 1, device=device)

    def save_samples(step_tag: str):
        G.eval()
        with torch.no_grad():
            fake = G(fixed_z)
            grid = make_grid(fake, nrow=8, normalize=True, value_range=(-1, 1))
            save_image(grid, sample_dir / f"samples_{step_tag}.png")
        G.train()

    global_step = 0
    save_samples("start")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss_d_accum = 0.0
        loss_g_accum = 0.0

        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device)

            # Filter: keep only REAL images for D's "real" batch
            real_mask = (y == 1)  # our dataset labels real=1, spoof=0
            if real_mask.sum() < 2:
                continue
            real_imgs = x[real_mask]

            bsz = real_imgs.size(0)
            z = torch.randn(bsz, z_dim, 1, 1, device=device)
            fake_imgs = G(z).detach()

            # --------------------
            # Train D
            # --------------------
            D.zero_grad(set_to_none=True)

            logits_real = D(real_imgs)
            logits_fake = D(fake_imgs)

            y_real = torch.ones_like(logits_real)
            y_fake = torch.zeros_like(logits_fake)

            loss_d = bce(logits_real, y_real) + bce(logits_fake, y_fake)
            loss_d.backward()
            opt_d.step()

            # --------------------
            # Train G (fool D + fool P)
            # --------------------
            G.zero_grad(set_to_none=True)
            z = torch.randn(bsz, z_dim, 1, 1, device=device)
            gen = G(z)

            logits_gen = D(gen)
            loss_g_adv = bce(logits_gen, torch.ones_like(logits_gen))

            # fool predictor: make predictor classify generated as REAL
            pred_in = to_predictor_input(gen, mean, std)
            logits_p = predictor(pred_in)  # shape (B,2)
            target_real = torch.full((bsz,), real_index, device=device, dtype=torch.long)
            loss_fool = ce(logits_p, target_real)

            loss_g = loss_g_adv + lam_fool * loss_fool
            loss_g.backward()
            opt_g.step()

            loss_d_accum += loss_d.item()
            loss_g_accum += loss_g.item()
            global_step += 1

            if global_step % 200 == 0:
                save_samples(f"e{epoch}_s{global_step}")

        dt = time.time() - t0
        loss_d_avg = loss_d_accum / max(1, len(loader))
        loss_g_avg = loss_g_accum / max(1, len(loader))
        print(f"[{epoch:02d}/{epochs}] lossD={loss_d_avg:.4f} lossG={loss_g_avg:.4f} ({dt:.1f}s)")

        # save checkpoints
        torch.save({"G": G.state_dict(), "D": D.state_dict(), "img_size": img_size, "z_dim": z_dim},
                   out_dir / "gan_last.pt")

    save_samples("end")
    print("✅ GAN training complete.")
    print("Samples saved to:", sample_dir.resolve())


if __name__ == "__main__":
    main()
