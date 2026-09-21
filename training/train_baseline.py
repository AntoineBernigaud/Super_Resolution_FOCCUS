"""Stage 1: deterministic conditional mean mu(x).

Serves three purposes, all of them needed before the diffusion model means anything:
  * it is the baseline any generative result must beat;
  * it carries the large scales, so the diffusion stage only has to generate short-
    correlation texture -- which is what makes patch-tiled sampling safe;
  * it validates the data pipeline end to end.

Masked Huber loss over finite ssha only.  Huber rather than MSE because the target
retains genuine L3 artifacts; combined with the +-0.5 m clip applied in data.py this
is belt and braces, and cheap.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config as C
from data import FullFieldDataset, load_stats
from nets import BaselineNet, block_repeat


def masked_huber(pred, target, mask, delta=1.0):
    d = pred - target
    a = d.abs()
    loss = torch.where(a <= delta, 0.5 * d * d, delta * (a - 0.5 * delta))
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def masked_rmse(pred, target, mask):
    return (((pred - target) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)).sqrt()


def lr_at(step, total, base_lr, warmup):
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1 + np.cos(np.pi * min(p, 1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=2,
                    help="whole 1024x912 fields per step")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--fine-width", type=int, default=64)
    ap.add_argument("--base", default="block",
                    choices=["block", "bicubic", "bilinear"],
                    help="residual base: piecewise constant or smooth")
    ap.add_argument("--up", default="pixelshuffle",
                    choices=["pixelshuffle", "resize"],
                    help="upsampler; resize-conv avoids checkerboard")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--target-cache", default="cache_ssha_wh.npy",
                    help="alternative ssha cache to TRAIN against "
                         "(e.g. a low-passed target); scoring always "
                         "uses the raw one")
    ap.add_argument("--out", default="runs/baseline")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    mean, std = load_stats()
    print(json.dumps(vars(args), indent=2))
    print(f"normalisation: mean {mean:+.6f}  std {std:.6f} m")

    # Stage 1 trains on WHOLE fields, not patches.  Two reasons, both measured by
    # smoke_test.py's receptive-field probe, which reports 128x304 coarse cells --
    # the entire grid:
    #   * the convolutional stack alone reaches 1 + 13*2 = 27 coarse cells, already
    #     more than twice a patch's 12-cell height, so a patch-trained net would
    #     lean on zero padding that is absent at full-field inference;
    #   * GroupNorm normalises over the spatial dimensions, so the dependence is in
    #     fact global -- the group statistics of a 96x96 patch are not those of a
    #     1024x912 field, and no amount of convolutional margin would fix that.
    # Whole fields are cheap here because the net is deterministic, and they make
    # training and inference identical.  (Stage 2 keeps patches: its tiles are the
    # same 96x96 at training and at sampling time, so its statistics match.)
    tr = FullFieldDataset("train", ssha_cache=args.target_cache)
    va = FullFieldDataset("val", ssha_cache=args.target_cache)
    print(f"[train] {len(tr)} days   [val] {len(va)} days")
    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, drop_last=True,
                       pin_memory=True, persistent_workers=args.workers > 0)
    dl_va = DataLoader(va, batch_size=args.batch, shuffle=False,
                       num_workers=args.workers, pin_memory=True,
                       persistent_workers=args.workers > 0)

    net = BaselineNet(args.width, args.fine_width,
                      base=args.base, up=args.up).to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"BaselineNet {n_par/1e6:.2f} M parameters")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    total = args.epochs * len(dl_tr)
    warmup = min(1000, total // 20)
    step, best = 0, float("inf")
    hist = []

    for ep in range(args.epochs):
        net.train()
        t0, run = time.time(), 0.0
        for bi, b in enumerate(dl_tr):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, args.lr, warmup)
            x = b["x"].to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = net(x)
            pred = pred.float()
            y = b["y"].to(dev, non_blocking=True)
            m = b["mask"].to(dev, non_blocking=True)
            loss = masked_huber(pred, y, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            run += loss.item()
            step += 1
            if bi % 200 == 0:
                print(f"  ep{ep} {bi}/{len(dl_tr)} loss {loss.item():.5f} "
                      f"lr {opt.param_groups[0]['lr']:.2e}", flush=True)

        # --- validation, reported in metres --------------------------------------
        net.eval()
        se_net = se_triv = n_pix = 0.0
        with torch.no_grad():
            for b in dl_va:
                x = b["x"].to(dev, non_blocking=True)
                y = b["y"].to(dev, non_blocking=True)
                m = b["mask"].to(dev, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = net(x).float()
                triv = block_repeat(x[:, :1])
                se_net += (((pred - y) ** 2) * m).sum().item()
                se_triv += (((triv - y) ** 2) * m).sum().item()
                n_pix += m.sum().item()
        rmse = np.sqrt(se_net / n_pix) * std
        rmse_triv = np.sqrt(se_triv / n_pix) * std
        hist.append(dict(epoch=ep, train_loss=run / len(dl_tr),
                         val_rmse_m=rmse, trivial_rmse_m=rmse_triv))
        extra = ""
        print(f"[ep {ep}] train {run/len(dl_tr):.5f}  "
              f"val RMSE {rmse*100:.3f} cm  "
              f"(block-repeat sla: {rmse_triv*100:.3f} cm, "
              f"skill {100*(1-rmse/rmse_triv):+.1f}%)"
              + extra + f"  {time.time()-t0:.0f}s", flush=True)

        sel = rmse
        if sel < best:
            best = sel
            torch.save({"model": net.state_dict(), "args": vars(args),
                        "val_rmse_m": rmse, "epoch": ep}, out / "best.pt")
            print(f"  saved best ({rmse*100:.3f} cm)")
        (out / "history.json").write_text(json.dumps(hist, indent=2))

    print(f"done. best val RMSE {best*100:.3f} cm -> {out/'best.pt'}")


if __name__ == "__main__":
    main()
