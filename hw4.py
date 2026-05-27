import os
import argparse
import random
import math
import csv
from glob import glob
from tqdm import tqdm

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = \
    "expandable_segments:True"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_image(path):
    img = Image.open(path).convert("RGB")
    return img


def pil_to_tensor(img):
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def tensor_to_uint8_chw(x):
    x = x.detach().cpu().clamp(0, 1)
    x = (x * 255.0).round().byte().numpy()
    return x


def calc_psnr(pred, target):
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    mse = F.mse_loss(pred, target)
    if mse.item() == 0:
        return 100.0
    return 10 * math.log10(1.0 / mse.item())


# ============================================================
# Dataset
# ============================================================

class RestorationDataset(Dataset):
    def __init__(
        self,
        root="./hw4_dataset/hw4_dataset",
        patch_size=128,
        train=True
    ):
        self.patch_size = patch_size
        self.train = train

        degraded_dir = os.path.join(
            root,
            "train",
            "degraded"
        )

        clean_dir = os.path.join(
            root,
            "train",
            "clean"
        )

        self.pairs = []

        degraded_files = sorted(
            glob(
                os.path.join(
                    degraded_dir,
                    "*.png"
                )
            )
        )

        for d_path in degraded_files:

            name = os.path.basename(d_path)

            # rain-xxx.png
            if name.startswith("rain-"):

                idx = name.replace(
                    "rain-",
                    ""
                ).replace(
                    ".png",
                    ""
                )

                clean_name = f"rain_clean-{idx}.png"

            # snow-xxx.png
            elif name.startswith("snow-"):

                idx = name.replace(
                    "snow-",
                    ""
                ).replace(
                    ".png",
                    ""
                )

                clean_name = f"snow_clean-{idx}.png"

            else:
                continue

            c_path = os.path.join(
                clean_dir,
                clean_name
            )

            if os.path.exists(c_path):
                self.pairs.append(
                    (
                        d_path,
                        c_path
                    )
                )

        print(
            f"Total pairs: {len(self.pairs)}"
        )

    def __len__(self):
        return len(self.pairs)

    def random_crop(self, x, y):
        _, h, w = x.shape
        ps = self.patch_size
        if h < ps or w < ps:

            new_h = max(h, ps)
            new_w = max(w, ps)

            x = TF.resize(
                x,
                [new_h, new_w]
            )

            y = TF.resize(
                y,
                [new_h, new_w]
            )

            _, h, w = x.shape

        top = random.randint(
            0,
            h - ps
        )

        left = random.randint(
            0,
            w - ps
        )

        x = x[
            :,
            top:top+ps,
            left:left+ps
        ]

        y = y[
            :,
            top:top+ps,
            left:left+ps
        ]

        return x, y

    def augment(self, x, y):

        if random.random() < 0.5:

            x = torch.flip(
                x,
                dims=[2]
            )

            y = torch.flip(
                y,
                dims=[2]
            )

        if random.random() < 0.5:

            x = torch.flip(
                x,
                dims=[1]
            )

            y = torch.flip(
                y,
                dims=[1]
            )

        k = random.randint(
            0,
            3
        )

        x = torch.rot90(
            x,
            k,
            [1, 2]
        )

        y = torch.rot90(
            y,
            k,
            [1, 2]
        )

        return x, y

    def __getitem__(self, idx):

        d_path, c_path = self.pairs[idx]

        degraded = pil_to_tensor(
            load_image(d_path)
        )

        clean = pil_to_tensor(
            load_image(c_path)
        )

        if self.train:

            degraded, clean = \
                self.random_crop(
                    degraded,
                    clean
                )

            degraded, clean = \
                self.augment(
                    degraded,
                    clean
                )

        return degraded, clean


class TestDataset(Dataset):

    def __init__(
        self,
        root="./hw4_dataset/hw4_dataset"
    ):

        self.test_dir = os.path.join(
            root,
            "test",
            "degraded"
        )

        self.paths = sorted(
            glob(
                os.path.join(
                    self.test_dir,
                    "*.png"
                )
            ),
            key=lambda x:
            int(
                os.path.basename(
                    x
                ).replace(
                    ".png",
                    ""
                )
            )
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        path = self.paths[idx]

        img = pil_to_tensor(
            load_image(path)
        )

        name = os.path.basename(path)

        return img, name


# ============================================================
# PromptIR-style Model
# ============================================================

class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class SimpleAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)

        q = F.normalize(q, dim=2)
        k = F.normalize(k, dim=2)

        attn = torch.matmul(q.transpose(-2, -1), k)
        attn = attn * self.temperature
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v.transpose(-2, -1))
        out = out.transpose(-2, -1)
        out = out.reshape(b, c, h, w)

        out = self.project_out(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, expansion=2.66):
        super().__init__()
        hidden = int(dim * expansion)

        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            kernel_size=3,
            padding=1,
            groups=hidden * 2
        )
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = SimpleAttention(dim, heads)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = FeedForward(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class PromptBlock(nn.Module):
    def __init__(self, dim, prompt_len=5, prompt_size=32):
        super().__init__()

        self.prompt_len = prompt_len
        self.prompt_size = prompt_size

        self.prompt_param = nn.Parameter(
            torch.randn(1, prompt_len, dim, prompt_size, prompt_size)
        )

        self.linear = nn.Linear(dim, prompt_len)

        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape

        emb = F.adaptive_avg_pool2d(x, 1).view(b, c)
        weights = self.linear(emb)
        weights = F.softmax(weights, dim=1)

        prompt = self.prompt_param.repeat(b, 1, 1, 1, 1)
        weights = weights.view(b, self.prompt_len, 1, 1, 1)

        prompt = torch.sum(prompt * weights, dim=1)
        prompt = F.interpolate(
            prompt,
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )

        out = self.conv(prompt)
        return x + out


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class PromptIR(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=32,
        num_blocks=[2, 2, 4, 4],
        heads=[1, 2, 4, 8]
    ):
        super().__init__()

        self.patch_embed = nn.Conv2d(
            inp_channels,
            dim,
            kernel_size=3,
            padding=1
        )

        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim, heads[0]) for _ in range(num_blocks[0])]
        )
        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim * 2,
                    heads[1]
                )
                for _ in range(
                    num_blocks[1]
                )
            ]
        )
        self.down2_3 = Downsample(dim * 2)

        self.encoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim * 4,
                    heads[2]
                )
                for _ in range(
                    num_blocks[2]
                )
            ]
        )
        self.down3_4 = Downsample(dim * 4)

        self.latent = nn.Sequential(
            *[
                TransformerBlock(
                    dim * 8,
                    heads[3]
                )
                for _ in range(
                    num_blocks[3]
                )
            ]
        )

        self.prompt3 = PromptBlock(dim * 4)
        self.prompt2 = PromptBlock(dim * 2)
        self.prompt1 = PromptBlock(dim)

        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1)

        self.decoder_level3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim * 4,
                    heads[2]
                )
                for _ in range(
                    num_blocks[2]
                )
            ]
        )

        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1)

        self.decoder_level2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim * 2,
                    heads[1]
                )
                for _ in range(
                    num_blocks[1]
                )
            ]
        )

        self.up2_1 = Upsample(dim * 2)
        self.reduce_chan_level1 = nn.Conv2d(dim * 2, dim, kernel_size=1)

        self.decoder_level1 = nn.Sequential(
            *[
                TransformerBlock(
                    dim,
                    heads[0]
                )
                for _ in range(
                    num_blocks[0]
                )
            ]
        )

        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (8 - h % 8) % 8
        mod_pad_w = (8 - w % 8) % 8
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode="reflect")
        return x, h, w

    def forward(self, inp_img):
        x, ori_h, ori_w = self.check_image_size(inp_img)

        inp_enc_level1 = self.patch_embed(x)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], dim=1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        inp_dec_level3 = self.prompt3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        inp_dec_level2 = self.prompt2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        inp_dec_level1 = self.prompt1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        residual = self.output(out_dec_level1)

        restored = x + residual
        restored = restored[:, :, :ori_h, :ori_w]

        return restored.clamp(0, 1)


# ============================================================
# Loss
# ============================================================

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()


def ssim_loss(x, y):
    """Small differentiable SSIM loss for images in [0, 1]."""
    x = x.clamp(0, 1)
    y = y.clamp(0, 1)

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    mu_y = F.avg_pool2d(y, kernel_size=3, stride=1, padding=1)

    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y

    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) *
        (sigma_x + sigma_y + c2)
    )

    return 1.0 - ssim.mean()


class CombinedRestorationLoss(nn.Module):
    """Charbonnier loss + optional SSIM loss."""
    def __init__(self, ssim_weight=0.1):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.ssim_weight = ssim_weight

    def forward(self, pred, target):
        loss_char = self.charbonnier(pred, target)
        if self.ssim_weight <= 0:
            return loss_char
        loss_ssim = ssim_loss(pred, target)
        return loss_char + self.ssim_weight * loss_ssim


class ModelEMA:
    def __init__(self, model, decay=0.999):
        import copy
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        ema_state = self.module.state_dict()
        model_state = model.state_dict()

        for k in ema_state.keys():
            if ema_state[k].dtype.is_floating_point:
                ema_state[k].mul_(self.decay).add_(
                    model_state[k].detach(),
                    alpha=1.0 - self.decay
                )
            else:
                ema_state[k].copy_(model_state[k])


# ============================================================
# Train
# ============================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_dataset = RestorationDataset(
        root=args.data_root,
        patch_size=args.patch_size,
        train=True
    )

    val_dataset = RestorationDataset(
        root=args.data_root,
        patch_size=args.patch_size,
        train=False
    )
    val_dataset.pairs = train_dataset.pairs

    num_samples = len(train_dataset)
    train_size = int(num_samples * 0.9)

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(num_samples, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_set = torch.utils.data.Subset(train_dataset, train_indices)
    val_set = torch.utils.data.Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )

    model = PromptIR(
        dim=args.dim,
        num_blocks=args.num_blocks,
        heads=args.heads
    ).to(device)

    num_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )
    print(f"Trainable parameters: {num_params / 1e6:.2f}M")
    print(f"Model blocks: {args.num_blocks}, heads: {args.heads}")

    criterion = CombinedRestorationLoss(
        ssim_weight=args.ssim_weight
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )

    os.makedirs(args.save_dir, exist_ok=True)
    log_file = os.path.join(
        args.save_dir,
        "train_log.csv"
    )
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "val_psnr", "learning_rate"])

    best_psnr = -1
    start_epoch = 1

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        best_psnr = ckpt.get("best_psnr", best_psnr)
        start_epoch = ckpt.get("epoch", start_epoch)

        print(
            f"Resumed from {args.resume}, "
            f"start epoch: {start_epoch}, "
            f"best PSNR: {best_psnr:.4f}"
        )

    ema = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs}",
            leave=False
        )

        for degraded, clean in pbar:

            degraded = degraded.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():

                restored = model(degraded)

                loss = criterion(
                    restored,
                    clean
                )

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            scaler.step(optimizer)
            scaler.update()

            if ema is not None:
                ema.update(model)

            total_loss += loss.item()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}"
            )

        avg_loss = total_loss / len(train_loader)

        eval_model = ema.module if ema is not None else model
        eval_model.eval()
        val_psnr = 0

        with torch.no_grad():
            val_bar = tqdm(
                val_loader,
                desc="Validation",
                leave=False,
                dynamic_ncols=True
            )

            for degraded, clean in val_bar:

                degraded = degraded.to(device)
                clean = clean.to(device)

                with torch.cuda.amp.autocast():
                    restored = eval_model(degraded)

                psnr = calc_psnr(
                    restored,
                    clean
                )

                val_psnr += psnr

                val_bar.set_postfix(
                    psnr=f"{psnr:.2f}"
                )

        val_psnr /= len(val_loader)

        current_lr = optimizer.param_groups[0]["lr"]

        with open(
            log_file,
            "a",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                [
                    epoch,
                    avg_loss,
                    val_psnr,
                    current_lr
                ]
            )

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Loss: {avg_loss:.6f} "
            f"Val PSNR: {val_psnr:.4f} "
            f"Learning Rate: {current_lr:.2e}"
        )

        scheduler.step()

        last_path = os.path.join(args.save_dir, "last.pth")
        torch.save(
            {
                "epoch": epoch,
                "model": (
                    ema.module.state_dict()
                    if ema is not None
                    else model.state_dict()
                ),
                "optimizer": optimizer.state_dict(),
                "best_psnr": best_psnr
            },
            last_path
        )

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_path = os.path.join(args.save_dir, "best.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model": (
                        ema.module.state_dict()
                        if ema is not None
                        else model.state_dict()
                    ),
                    "optimizer": optimizer.state_dict(),
                    "best_psnr": best_psnr
                },
                best_path
            )
            print(f"Saved best model. PSNR = {best_psnr:.4f}")


def tile_predict(model, img, tile_size=128, overlap=24):
    """Tiled inference to avoid whole-image attention OOM."""
    _, _, h, w = img.shape
    stride = tile_size - overlap

    output = torch.zeros_like(img)
    weight = torch.zeros_like(img)

    y_positions = list(range(0, h, stride))
    x_positions = list(range(0, w, stride))

    for y in y_positions:
        for x in x_positions:
            y1 = min(y, max(h - tile_size, 0))
            x1 = min(x, max(w - tile_size, 0))
            y2 = min(y1 + tile_size, h)
            x2 = min(x1 + tile_size, w)

            patch = img[:, :, y1:y2, x1:x2]

            with torch.cuda.amp.autocast():
                pred = model(patch)

            output[:, :, y1:y2, x1:x2] += pred
            weight[:, :, y1:y2, x1:x2] += 1

    return output / weight.clamp_min(1e-6)


def tta_predict(model, img, tile_size=128, overlap=24):
    """4-way test-time augmentation."""
    transforms = [
        lambda x: x,
        lambda x: torch.flip(x, [3]),
        lambda x: torch.flip(x, [2]),
        lambda x: torch.rot90(x, 1, [2, 3]),
    ]

    inverse = [
        lambda x: x,
        lambda x: torch.flip(x, [3]),
        lambda x: torch.flip(x, [2]),
        lambda x: torch.rot90(x, -1, [2, 3]),
    ]

    preds = []
    for t, inv in zip(transforms, inverse):
        x = t(img)
        y = tile_predict(
            model,
            x,
            tile_size=tile_size,
            overlap=overlap
        )
        preds.append(inv(y))

    return torch.mean(torch.stack(preds, dim=0), dim=0)

# ============================================================
# Predict
# ============================================================


def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = TestDataset(args.data_root)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = PromptIR(
        dim=args.dim,
        num_blocks=args.num_blocks,
        heads=args.heads
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    results = {}

    with torch.no_grad():
        for img, name in loader:
            img = img.to(device)
            if args.use_tta:
                restored = tta_predict(
                    model,
                    img,
                    tile_size=args.tile_size,
                    overlap=args.overlap
                )
            else:
                restored = tile_predict(
                    model,
                    img,
                    tile_size=args.tile_size,
                    overlap=args.overlap
                )

            arr = tensor_to_uint8_chw(restored[0])
            results[name[0]] = arr

            print("Processed:", name[0], arr.shape, arr.dtype)

    np.savez(args.output, **results)
    print(f"Saved prediction file to {args.output}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, required=True,
                        choices=["train", "predict"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best.pth")
    parser.add_argument("--output", type=str, default="pred.npz")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=128)

    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--ssim_weight", type=float, default=0.1)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--tile_size", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=24)
    parser.add_argument("--use_tta", action="store_true")

    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Stronger default architecture for Colab/A100.
    # Keep train and predict exactly consistent.
    args.num_blocks = [2, 2, 4, 6]
    args.heads = [1, 2, 4, 8]

    set_seed(args.seed)

    if args.mode == "train":
        train(args)
    elif args.mode == "predict":
        predict(args)


if __name__ == "__main__":
    main()
