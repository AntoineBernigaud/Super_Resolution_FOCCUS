"""Stage 2: EDM diffusion over the residual r = y - mu(x).

Why the residual and not ssha directly:
  * r is near-zero-mean, roughly stationary texture, so observing it on 25% of pixels
    still samples its distribution fairly;
  * 0 is r's conditional mean, so filling unobserved pixels with 0 is not the
    physically-loaded zero-fill the dataset contract warns against;
  * r has a short correlation length, which is what makes tiled sampling of the full
    field safe (see edm.sample_field).

Known approximation: the noised input still contains filled pixels, so the network
sees a train/inference mismatch near mask boundaries.  Mitigated by the coverage
filter and by the mask channel.  Ambient diffusion was tried as the principled fix
and did not help -- it cost 1.1% val loss and left the 25 km phase coupling
unchanged -- so the mismatch stands as a known approximation.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

import config as C
import edm
from data import PatchDataset, load_stats
from nets import DiffusionUNet


class EMA:
    """Diffusion sample quality depends on it far more than on the raw weights."""

    def __init__(self, model, decay=0.9995):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.lerp_(p.detach(), 1 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)


def lr_at(step, total, base_lr, warmup):
    """Warmup then cosine decay.

    The first run of this script held LR constant after warmup -- an oversight, not a
    decision -- and its validation loss bottomed at epoch 9 of 60, then degraded 23%
    while train fell 37%.  The schedule now completes near where it actually peaks.
    """
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1 + np.cos(np.pi * min(p, 1.0)))


def residual_scale(loader, n_batches=200):
    """Std of the observed residual over the train split.

    EDM assumes sigma_data = 1, so the residual is divided by this before any noise
    is added.  Estimated from a sample of batches rather than a full pass: 200
    batches x 32 x 96 x 96 x ~0.6 coverage is ~35 M pixels, plenty for a std.
    """
    s = ss = n = 0.0
    for k, b in enumerate(loader):
        if k >= n_batches:
            break
        r = ((b["y"] - b["mu"]) * b["mask"]).double()
        m = b["mask"].double()
        s += r.sum().item()
        ss += (r * r).sum().item()
        n += m.sum().item()
    mean = s / n
    return float(np.sqrt(max(ss / n - mean * mean, 1e-12))), mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20,
                    help="was 60 until 2026-09-16.  Every stage 2 ever trained here "
                         "bottoms at epoch 7-12 and then overfits 15-36%, and the "
                         "regularisation sweep showed 20 epochs reaches the same best "
                         "val as 60 (0.07005 vs 0.07002) in 1h09 against 3h20.  The "
                         "cosine schedule now completes near where the run peaks, "
                         "which is what it was written for.")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--cov", type=float, default=0.5)
    ap.add_argument("--ch", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--mu", default=str(C.ROOT / "mu_whitened.npy"))
    ap.add_argument("--target-cache", default="cache_ssha_wh.npy",
                    help="alternative ssha cache to TRAIN against "
                         "(e.g. a low-passed target); scoring always "
                         "uses the raw one")
    ap.add_argument("--out", default="runs/diffusion_whitened")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    mean, std = load_stats()
    print(f"sigma distribution: P_MEAN {edm.P_MEAN} P_STD {edm.P_STD} "
          f"SIGMA_DATA {edm.SIGMA_DATA}")
    print(json.dumps(vars(args), indent=2))

    tr = PatchDataset("train", cov_thresh=args.cov, with_mu=True, mu_path=args.mu,
                      ssha_cache=args.target_cache, balance=True)
    va = PatchDataset("val", cov_thresh=args.cov, with_mu=True, mu_path=args.mu,
                      ssha_cache=args.target_cache, balance=False)
    sampler = WeightedRandomSampler(
        torch.as_tensor(tr.weights, dtype=torch.double), len(tr), replacement=True)
    dl_tr = DataLoader(tr, batch_size=args.batch, sampler=sampler,
                       num_workers=args.workers, drop_last=True, pin_memory=True,
                       persistent_workers=args.workers > 0)
    dl_va = DataLoader(va, batch_size=args.batch, shuffle=False,
                       num_workers=args.workers, pin_memory=True,
                       persistent_workers=args.workers > 0)

    scale_file = out / "residual_scale.json"
    if scale_file.exists():
        r_scale = json.loads(scale_file.read_text())["r_scale"]
    else:
        r_scale, r_mean = residual_scale(dl_tr)
        scale_file.write_text(json.dumps({"r_scale": r_scale, "r_mean": r_mean}))
        print(f"residual: mean {r_mean:+.5f}  std {r_scale:.5f} (normalised units)"
              f"  = {r_scale*std*100:.3f} cm")
    print(f"r_scale {r_scale:.5f}")

    net = DiffusionUNet(tuple(args.ch), dropout=args.dropout).to(dev)
    print(f"DiffusionUNet {sum(p.numel() for p in net.parameters())/1e6:.2f} M params")
    ema = EMA(net, args.ema)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)

    total = args.epochs * len(dl_tr)
    warmup = min(2000, total // 20)
    step, best, hist = 0, float("inf"), []

    def prep(b):
        y = b["y"].to(dev, non_blocking=True)
        mu = b["mu"].to(dev, non_blocking=True)
        m = b["mask"].to(dev, non_blocking=True)
        x = b["x"].to(dev, non_blocking=True)
        r0 = ((y - mu) * m) / r_scale        # 0 exactly where unobserved
        return r0, m, mu, x

    for ep in range(args.epochs):
        net.train()
        t0, run = time.time(), 0.0
        for bi, b in enumerate(dl_tr):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, args.lr, warmup)
            r0, m, mu, x = prep(b)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = edm.edm_loss(net, r0, m, mu, x, obs_mask=m)
            opt.zero_grad(set_to_none=True)
            loss.float().backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            ema.update(net)
            run += loss.item()
            step += 1
            if bi % 200 == 0:
                print(f"  ep{ep} {bi}/{len(dl_tr)} loss {loss.item():.4f}",
                      flush=True)

        # Validation uses the EMA weights and a FIXED noise seed, so epoch-to-epoch
        # differences are the model changing rather than the sigma draw.
        ema.shadow.eval()
        vl, nb = 0.0, 0
        g = torch.Generator(device="cpu").manual_seed(1234)
        with torch.no_grad():
            for b in dl_va:
                r0, m, mu, x = prep(b)
                torch.manual_seed(int(torch.randint(0, 2**31, (1,), generator=g)))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    vl += edm.edm_loss(ema.shadow, r0, m, mu, x,
                                       obs_mask=m).item()
                nb += 1
        vl /= max(nb, 1)
        hist.append(dict(epoch=ep, train=run / len(dl_tr), val=vl))
        print(f"[ep {ep}] train {run/len(dl_tr):.4f}  val {vl:.4f}  "
              f"{time.time()-t0:.0f}s", flush=True)

        if vl < best:
            best = vl
            torch.save({"model": net.state_dict(), "ema": ema.shadow.state_dict(),
                        "args": vars(args), "r_scale": r_scale, "epoch": ep,
                        "val": vl}, out / "best.pt")
            print(f"  saved best ({vl:.4f})")
        torch.save({"model": net.state_dict(), "ema": ema.shadow.state_dict(),
                    "args": vars(args), "r_scale": r_scale, "epoch": ep},
                   out / "last.pt")
        (out / "history.json").write_text(json.dumps(hist, indent=2))

    print(f"done. best val {best:.4f}")


if __name__ == "__main__":
    main()
