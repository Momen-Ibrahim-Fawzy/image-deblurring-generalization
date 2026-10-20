"""Training and evaluation loops."""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from deblur.losses import DeblurLoss
from deblur.metrics import RunningMetrics


def _to_device(batch: dict, device: str) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: str = "cuda", *,
             y_channel: bool = False, save_dir: str | Path | None = None,
             csv_path: str | Path | None = None, amp: bool = True) -> dict[str, float]:
    """Runs the model over a test set and returns mean PSNR/SSIM (optionally saving outputs)."""
    model.eval().to(device)
    metrics = RunningMetrics(y_channel=y_channel)
    save_dir = Path(save_dir) if save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    for batch in loader:
        batch = _to_device(batch, device)
        with torch.autocast(device_type="cuda", enabled=amp and device.startswith("cuda")):
            pred = model(batch["blur"], batch.get("masks"))
        pred = pred.float().clamp(0, 1)
        metrics.update(batch["name"], pred, batch["sharp"])
        if save_dir:
            for name, img in zip(batch["name"], pred):
                arr = (img.permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).astype("uint8")
                Image.fromarray(arr).save(save_dir / f"{name}.png")

    if csv_path:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "psnr", "ssim"])
            writer.writerows(metrics.records)
    return metrics.summary()


def train(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader | None, *,
          iterations: int, lr: float, out_dir: str | Path, device: str = "cuda",
          freq_weight: float = 0.1, val_every: int = 5000, log_every: int = 100,
          backbone_lr_scale: float = 0.1, grad_clip: float = 0.01) -> None:
    """Fine-tunes a (SAM-guided) restorer with AdamW + cosine schedule.

    The pre-trained backbone gets a smaller learning rate than the new fusion
    layer, so fine-tuning adapts to the prior without destroying the backbone.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to(device).train()

    backbone_params = list(model.backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr * backbone_lr_scale},
        {"params": head_params, "lr": lr},
    ], betas=(0.9, 0.9), weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations, eta_min=1e-6)
    criterion = DeblurLoss(freq_weight)
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))

    best_psnr, step, t0 = -math.inf, 0, time.time()
    while step < iterations:
        for batch in train_loader:
            batch = _to_device(batch, device)
            with torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
                pred = model(batch["blur"], batch.get("masks"))
                losses = criterion(pred.float(), batch["sharp"])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1

            if step % log_every == 0:
                print(f"[{step:>7}/{iterations}] loss={losses['loss'].item():.4f} "
                      f"l1={losses['l1'].item():.4f} freq={losses['freq'].item():.4f} "
                      f"lr={scheduler.get_last_lr()[-1]:.2e} {time.time() - t0:.0f}s")
            if val_loader is not None and (step % val_every == 0 or step == iterations):
                result = evaluate(model, val_loader, device)
                model.train()
                print(f"  val psnr={result['psnr']:.3f} ssim={result['ssim']:.4f}")
                if result["psnr"] > best_psnr:
                    best_psnr = result["psnr"]
                    torch.save({"params": model.state_dict(), "step": step, "val": result},
                               out_dir / "best.pth")
            if step % val_every == 0:
                torch.save({"params": model.state_dict(), "step": step}, out_dir / "last.pth")
            if step >= iterations:
                break
