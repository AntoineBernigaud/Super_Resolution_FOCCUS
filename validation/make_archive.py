"""Sample the parent (DUACS-only) model over many test days and archive the result.

Both the FSS and the cross-scale-transfer analyses need an ensemble on a large number
of days.  Sampling dominates the cost (~45 s per full-field member), so it is done
once here and written to disk; the analyses then run in minutes and can be re-run
freely as the diagnostics evolve.

Stored in normalised units as float16 -- the same precision the cache uses, three
orders below anything these diagnostics resolve.
"""
import argparse
import numpy as np
import torch
from pathlib import Path

import config as C
import edm
from data import FullFieldDataset, load_stats
from nets import baseline_from_ckpt, BaselineNet, DiffusionUNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="runs/baseline_whitened/best.pt")
    ap.add_argument("--diffusion", default="runs/diffusion_whitened/best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--days", type=int, default=40)
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--tile-batch", type=int, default=64)
    ap.add_argument("--min-cov", type=float, default=0.15)
    ap.add_argument("--dates", nargs="*", default=None,
                    help="explicit dates (YYYY-MM-DD); overrides --split/--days")
    ap.add_argument("--center-residual", action="store_true", default=True,
                    help="subtract the sampled residual's field mean before adding it "
                         "to mu.  ON BY DEFAULT.  r is zero-mean BY CONSTRUCTION -- mu "
                         "is the conditional mean -- so any domain mean in it is "
                         "artifact, and the model has no skill at the domain mean "
                         "regardless (that is DUACS's job, carried in mu).  Measured "
                         "over 20 test days: member bias +0.892 -> +0.076 cm, rank "
                         "histogram ramp -> flat (mean rank 0.394 -> 0.467, ends ratio "
                         "0.99), CRPS 1.617 -> 1.537 cm.  Subtracting a scalar per "
                         "field cannot change a spectrum, and nothing else moved.")
    ap.add_argument("--no-center-residual", dest="center_residual",
                    action="store_false",
                    help="reproduce archives made before 2026-09-11, or measure the "
                         "raw offset")
    ap.add_argument("--sigma-max", type=float, default=13.0,
                    help="start the reverse process here instead of edm.SIGMA_MAX=80. "
                         "Sampling-side only, no retraining.  Training draws sigma>10 "
                         "0.18%% of the time, so the first 4-5 of 32 Karras steps run "
                         "through an effectively untrained network, and at large sigma "
                         "c_out -> sigma_data passes its output straight through.  "
                         "Measured on the 10-60 km band against the training target: "
                         "80 -> 2.21x, 20 -> 1.10x, 10 -> 0.48x.")
    ap.add_argument("--noise-mode", default="fixed",
                    choices=["independent", "fixed", "ar1"],
                    help="how each member's initial noise varies from day to day. "
                         "DEFAULT IS 'fixed'.  Archives made before 2026-09-09 used "
                         "'independent' -- pass it explicitly to reproduce them. "
                         "'independent' redraws it every day, so "
                         "the texture is temporally uncorrelated -- measured: the "
                         "member changes 1.93x as much per day as the truth and its "
                         "lag-1 correlation is 0.670 against the truth's 0.887. "
                         "'fixed' reuses one field per member for the whole record. "
                         "'ar1' correlates it with a decorrelation time --noise-tau; "
                         "each day is still exactly N(0,1) marginally, so no "
                         "single-day statistic changes.")
    ap.add_argument("--noise-tau", type=float, default=14.0,
                    help="ar1 decorrelation time in days")
    ap.add_argument("--out", default="archive_wh13")
    args = ap.parse_args()

    if args.sigma_max is not None:
        edm.SIGMA_MAX = args.sigma_max
        print(f"sampling schedule: sigma {edm.SIGMA_MIN} .. {edm.SIGMA_MAX}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    mean, std = load_stats()

    bck = torch.load(args.baseline, map_location="cpu", weights_only=False)
    base = baseline_from_ckpt(bck).to(dev).eval()
    base.load_state_dict(bck["model"])
    dck = torch.load(args.diffusion, map_location="cpu", weights_only=False)
    net = DiffusionUNet(tuple(dck["args"]["ch"])).to(dev).eval()
    net.load_state_dict(dck["ema"])
    r_scale = dck["r_scale"]
    print(f"baseline ep{bck['epoch']}  diffusion ep{dck['epoch']}  "
          f"r_scale {r_scale:.5f}")

    ds = FullFieldDataset(None if args.dates else args.split)
    cov = np.load(C.PATCH_INDEX)["day_cov"][ds.t]
    if args.dates:
        want = set(args.dates)
        pick = np.array([k for k in range(len(ds))
                         if str(ds.dates[ds.t[k]]) in want])
        missing = want - {str(ds.dates[ds.t[k]]) for k in pick}
        if missing:
            print(f"not in the record: {sorted(missing)}")
        print(f"{len(pick)} of {len(want)} requested dates found")
        _run(ds, pick, cov, base, net, r_scale, args, out)
        return
    usable = np.nonzero(cov > args.min_cov)[0]
    # Spread the sample evenly over the period rather than taking the best-covered
    # days, so the time mean is not biased toward one part of the season.
    pick = usable[np.linspace(0, len(usable) - 1, min(args.days, len(usable))
                              ).round().astype(int)]
    pick = np.unique(pick)
    print(f"{len(pick)} days of {len(ds)} in '{args.split}' (coverage > {args.min_cov})")
    _run(ds, pick, cov, base, net, r_scale, args, out)


def _run(ds, pick, cov, base, net, r_scale, args, out):
    import torch
    dev = torch.device("cuda")
    for n, k in enumerate(pick):
        b = ds[int(k)]
        date = str(ds.dates[ds.t[int(k)]])
        f = out / f"pred_{date}.npz"
        if f.exists():
            print(f"  [{n+1}/{len(pick)}] {date} exists, skipping", flush=True)
            continue
        x = b["x"][None].to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            mu = base(x).float()
        # AR(1)/fixed need the days in chronological order and state carried between
        # them; _run() iterates `pick` in the order given, which make_archive sorts.
        noise = getattr(args, "_noise", None)
        if noise is None:
            noise = args._noise = {}
        ens = []
        for e in range(args.members):
            g = torch.Generator(device=dev).manual_seed(7000 + 100 * int(k) + e)
            init = None
            if args.noise_mode != "independent":
                gf = torch.Generator(device=dev).manual_seed(4242 + e)
                if e not in noise:
                    noise[e] = torch.randn(mu.shape, device=dev, generator=gf)
                elif args.noise_mode == "ar1":
                    rho = float(np.exp(-1.0 / max(args.noise_tau, 1e-6)))
                    eps = torch.randn(mu.shape, device=dev, generator=g)
                    noise[e] = rho * noise[e] + np.sqrt(1.0 - rho ** 2) * eps
                init = noise[e]
            r = edm.sample_field(net, mu, x, n_steps=args.steps,
                                 tile_batch=args.tile_batch, generator=g,
                                 init_noise=init)
            if args.center_residual:
                r = r - r.mean()
            ens.append((mu + r_scale * r)[0, 0].cpu().numpy().astype(np.float16))
        np.savez(f, ens=np.stack(ens), mu=mu[0, 0].cpu().numpy().astype(np.float16),
                 t=int(ds.t[int(k)]), date=date, cov=float(cov[k]))
        print(f"  [{n+1}/{len(pick)}] {date}  cov {cov[k]:.3f}  -> {f.name}",
              flush=True)

    print("done")


if __name__ == "__main__":
    main()
