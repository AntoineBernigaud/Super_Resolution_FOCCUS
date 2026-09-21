"""Run the frozen stage-1 net over the whole record and cache mu on the fine grid.

The diffusion stage needs mu for every patch it draws.  Recomputing it each step
would dominate the step time, so it is computed once here into a float16 memmap
(828 x 1024 x 912 = 1.5 GB).  float16 costs ~1e-3 in normalised units, i.e. ~0.007 cm
of sea level -- far below anything that matters here.

BaselineNet is fully convolutional and uses GroupNorm, so applying it to the full
1024x912 field after training on 96x96 patches is exact, not an approximation.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import config as C
from data import FullFieldDataset, load_stats
from nets import BaselineNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/baseline_whitened/best.pt")
    ap.add_argument("--out", default=str(C.ROOT / "mu_whitened.npy"))
    ap.add_argument("--target-cache", default=None,
                    help="alternative ssha cache to TRAIN against "
                         "(e.g. a low-passed target); scoring always "
                         "uses the raw one")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing cache")
    args = ap.parse_args()

    # mu.npy was silently overwritten once by a later experiment, which left every
    # run that had trained against it pointing at a different field than the one it
    # learned on.  Nothing failed loudly; the mismatch only showed up when a
    # diagnostic was traced back to the checkpoint that produced the cache.  Name
    # the output per run, and refuse to clobber.
    if Path(args.out).exists() and not args.force:
        raise SystemExit(f"{args.out} exists -- pass --force, or use a per-run name")

    dev = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net = BaselineNet(ck["args"]["width"], ck["args"]["fine_width"],
                      base=ck["args"].get("base", "block"),
                      up=ck["args"].get("up", "pixelshuffle")).to(dev).eval()
    net.load_state_dict(ck["model"])
    print(f"loaded {args.ckpt}  (epoch {ck['epoch']}, "
          f"val RMSE {ck['val_rmse_m']*100:.3f} cm)")

    ds = FullFieldDataset()
    mean, std = load_stats()
    mm = np.lib.format.open_memmap(
        args.out, mode="w+", dtype=np.float16,
        shape=(len(ds), C.NLAT_F, C.NLON_F))

    se = n = 0.0
    with torch.no_grad():
        for k in range(len(ds)):
            b = ds[k]
            x = b["x"][None].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mu = net(x).float()
            mm[k] = mu[0, 0].cpu().numpy().astype(np.float16)
            m = b["mask"][None].to(dev)
            se += (((mu - b["y"][None].to(dev)) ** 2) * m).sum().item()
            n += m.sum().item()
            if k % 100 == 0:
                print(f"  {k}/{len(ds)}", flush=True)
    mm.flush()
    print(f"wrote {args.out}  ({Path(args.out).stat().st_size/1e9:.2f} GB)")
    print(f"full-record masked RMSE of mu: {np.sqrt(se/n)*std*100:.3f} cm")


if __name__ == "__main__":
    main()
