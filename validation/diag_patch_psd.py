"""Does the model reproduce the TARGET's spectrum?  And if not, is it the sampler?

A diffusion model trained on samples from p(r | mu, x) is supposed to produce samples
from p(r | mu, x).  Matching the target's spectrum is the defining property, not a
bonus, so the member running 2-3x hot between 60 and 10 km is a statement about this
implementation and not about the target.  Two things can break it:

  TRAINING     the learned score is wrong -- capacity, epochs, or the sigma
               distribution (P_MEAN/P_STD/RHO are Karras's ImageNet values here and
               have never been tuned).  In a diffusion model the noise level sets
               WHICH SPATIAL SCALE is being resolved, so the distribution over sigma
               is exactly the knob that allocates effort across the spectrum.

  SAMPLING     MultiDiffusion is a heuristic, not an exact sampler.  Averaging
               per-tile denoised estimates does not give the score of the global
               distribution, and a 96x96 tile is 167 x 159 km, so nothing above the
               tile can be generated coherently.  No amount of retraining fixes this.

This separates them.  The same days are sampled twice -- once tile-by-tile with
`sample_patch`, which is EXACTLY the configuration the network was trained in, and
once through `sample_field`, which is what every archive uses -- and both are scored
against the target residual with one identical estimator.

  patch / target ~ 1 and field / target >> 1  ->  the sampler, not the training
  both >> 1                                   ->  the training, i.e. hyperparameters
  both ~ 1                                    ->  the excess is mu's, not stage 2's

THE ESTIMATOR.  Along-COLUMN segments, because the fine grid's latitude spacing is
1.7394 km everywhere while the longitude spacing runs from 1.35 km at 74N to 1.71 km
at 64N -- an along-row estimator would smear a 26% range of wavelengths into each bin.
Runs of observed pixels are cut into fixed 64-point segments so every periodogram
shares one frequency axis, linearly detrended (a segment cannot represent the scales
it is shorter than, and leaving them in leaks into every bin) and Hann windowed.

Both model fields are complete, but they are masked to the TARGET's observed pixels
and passed through the same segment finder, so the three curves are built from
literally the same samples and any bias in the estimator divides out of the ratios.
`obs_mask` is left at None for both samplers -- the mask-channel mismatch is a
separate question, measured by `diag_bias.py`, and feeding it here would confound.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import config as C
import edm
from data import FullFieldDataset, load_stats, eval_box_slices
from nets import baseline_from_ckpt, DiffusionUNet

SEG = 64                                    # 64 * 1.7394 km = 111.3 km
DY_KM = C.DLAT_C / C.REFINE_LAT * 111.320


def seg_psd(fields, mask, acc, nseg):
    """Accumulate |FFT|^2 of every 64-point observed column run, for each field.

    `fields` is a dict name -> 2-D array.  The runs are found ONCE, on the mask, so
    every field contributes the identical samples.
    """
    win = np.hanning(SEG)
    wnorm = (win ** 2).sum()
    t = np.arange(SEG, dtype=np.float64)
    tc = t - t.mean()
    tss = (tc ** 2).sum()
    h, w = mask.shape
    for j in range(w):
        col = mask[:, j]
        if col.sum() < SEG:
            continue
        # maximal runs of observed pixels in this column
        d = np.diff(np.concatenate([[0], (col > 0).astype(np.int8), [0]]))
        starts = np.nonzero(d == 1)[0]
        ends = np.nonzero(d == -1)[0]
        for a, b in zip(starts, ends):
            for s0 in range(a, b - SEG + 1, SEG):
                ys = {k: f[s0:s0 + SEG, j].astype(np.float64)
                      for k, f in fields.items()}
                if not all(np.isfinite(v).all() for v in ys.values()):
                    continue
                for name, y in ys.items():
                    y = y - y.mean()
                    y = y - tc * (tc * y).sum() / tss        # linear detrend
                    F = np.fft.rfft(y * win)
                    acc[name] += (np.abs(F) ** 2) * (2.0 * DY_KM / wnorm)
                nseg[0] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="runs/baseline_whitened/best.pt")
    ap.add_argument("--diffusion", default="runs/diffusion_whitened/best.pt")
    ap.add_argument("--target-cache", default="cache_ssha_wh.npy",
                    help="the cache the model was TRAINED against -- comparing to "
                         "the raw truth instead would re-measure the artifact "
                         "rather than the model")
    ap.add_argument("--days", type=int, default=6)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--tile-batch", type=int, default=64)
    # Sampling-side schedule.  These cost nothing to change -- no retraining -- and
    # sigma_max is the one that decides whether the reverse process starts inside or
    # outside the range the network was actually trained on.
    ap.add_argument("--sigma-max", type=float, default=None)
    ap.add_argument("--sigma-min", type=float, default=None)
    ap.add_argument("--out", default="validation/plots/lam1.0/patch_psd")
    args = ap.parse_args()

    if args.sigma_max is not None:
        edm.SIGMA_MAX = args.sigma_max
    if args.sigma_min is not None:
        edm.SIGMA_MIN = args.sigma_min
    print(f"schedule: sigma {edm.SIGMA_MIN} .. {edm.SIGMA_MAX}, rho {edm.RHO}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    mean, std = load_stats()

    bck = torch.load(args.baseline, map_location="cpu", weights_only=False)
    base = baseline_from_ckpt(bck).to(dev).eval()
    base.load_state_dict(bck["model"])
    dck = torch.load(args.diffusion, map_location="cpu", weights_only=False)
    net = DiffusionUNet(tuple(dck["args"]["ch"]),
                        mask_channel=not dck["args"]["no_mask_channel"]).to(dev).eval()
    net.load_state_dict(dck["ema"])
    r_scale = dck["r_scale"]
    to_cm = r_scale * std * 100.0
    print(f"target {args.target_cache}   r_scale {r_scale:.5f} = {to_cm:.3f} cm")

    ds = FullFieldDataset("test", ssha_cache=args.target_cache)
    si, sj = eval_box_slices()
    idx = np.load(C.PATCH_INDEX)
    pick = np.argsort(-idx["day_cov"][ds.t])[:args.days]

    ph, pw = C.PATCH_F
    pch, pcw = C.PATCH_C
    oi_c, oj_c = C.patch_origins()
    # sample_field is left to its own full origin set -- that IS production
    # MultiDiffusion -- but the patch sampler only needs the tiles that get scored
    origins = [(i, j) for i in oi_c for j in oj_c
               if si.start <= i * C.REFINE_LAT
               and i * C.REFINE_LAT + ph <= si.stop
               and sj.start <= j * C.REFINE_LON
               and j * C.REFINE_LON + pw <= sj.stop]
    print(f"{len(origins)} scored tiles of {ph}x{pw} inside the eval box, "
          f"{args.steps} steps")

    names = ["target", "patch sampler", "MultiDiffusion"]
    acc = {n: np.zeros(SEG // 2 + 1) for n in names}
    nseg = [0]

    for c, k in enumerate(pick):
        b = ds[int(k)]
        date = str(ds.dates[ds.t[int(k)]])
        x = b["x"][None].to(dev)
        y = b["y"][None].to(dev)
        m = b["mask"][None].to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            mu = base(x).float()
        r0 = ((y - mu) * m) / r_scale

        g = torch.Generator(device=dev).manual_seed(1234 + c)
        r_md = edm.sample_field(net, mu, x, n_steps=args.steps,
                                tile_batch=args.tile_batch, generator=g)

        r0_np = r0[0, 0].cpu().numpy() * to_cm
        md_np = r_md[0, 0].cpu().numpy() * to_cm
        m_np = b["mask"][0].numpy() > 0

        # Score TILE BY TILE.  Patch-only sampling leaves the tile seams
        # discontinuous, and a 64-point column segment straddling one would inject
        # broadband power into the patch curve -- making it look hot for a reason
        # that has nothing to do with the learned distribution, i.e. faking the
        # very conclusion this diagnostic exists to test.  Keeping every segment
        # inside a single tile removes the seams from the comparison entirely.
        before = nseg[0]
        for st_ in range(0, len(origins), args.tile_batch):
            chunk = origins[st_:st_ + args.tile_batch]
            mt, xt = [], []
            for i, j in chunk:
                i0, j0 = i * C.REFINE_LAT, j * C.REFINE_LON
                mt.append(mu[0, :, i0:i0 + ph, j0:j0 + pw])
                xt.append(x[0, :, i:i + pch, j:j + pcw])
            d = edm.sample_patch(net, torch.stack(mt), torch.stack(xt),
                                 n_steps=args.steps, generator=g)
            d = d[:, 0].cpu().numpy() * to_cm
            for q, (i, j) in enumerate(chunk):
                i0, j0 = i * C.REFINE_LAT, j * C.REFINE_LON
                # only tiles wholly inside the eval box, so this measures the same
                # region every other diagnostic reports on
                if not (si.start <= i0 and i0 + ph <= si.stop
                        and sj.start <= j0 and j0 + pw <= sj.stop):
                    continue
                tsl = (slice(i0, i0 + ph), slice(j0, j0 + pw))
                seg_psd({"target": r0_np[tsl],
                         "patch sampler": d[q],
                         "MultiDiffusion": md_np[tsl]}, m_np[tsl], acc, nseg)
        print(f"  {c+1}/{len(pick)}  {date}  +{nseg[0]-before} segments", flush=True)

    n = max(nseg[0], 1)
    P = {k: v / n for k, v in acc.items()}
    freq = np.fft.rfftfreq(SEG, DY_KM)
    wl = np.where(freq > 0, 1.0 / np.maximum(freq, 1e-30), np.inf)

    print(f"\n{n} segments from {len(pick)} days\n")
    print(f"{'lambda km':>10}{'target':>12}{'patch':>12}{'MultiDiff':>12}"
          f"{'patch/tgt':>11}{'MD/tgt':>9}")
    rows = []
    for i in range(1, len(wl)):
        t_, p_, m_ = P["target"][i], P["patch sampler"][i], P["MultiDiffusion"][i]
        rows.append(dict(lambda_km=float(wl[i]), target=float(t_),
                         patch=float(p_), multidiff=float(m_)))
        print(f"{wl[i]:10.1f}{t_:12.4g}{p_:12.4g}{m_:12.4g}"
              f"{p_/max(t_,1e-30):11.3f}{m_/max(t_,1e-30):9.3f}")

    band = [i for i in range(1, len(wl)) if 10.0 <= wl[i] <= 60.0]
    bt = sum(P["target"][i] for i in band)
    bp = sum(P["patch sampler"][i] for i in band)
    bm = sum(P["MultiDiffusion"][i] for i in band)
    print(f"\n10-60 km band:  patch/target {bp/max(bt,1e-30):.3f}   "
          f"MultiDiffusion/target {bm/max(bt,1e-30):.3f}")
    print("patch ~1 and MD >1 -> the sampler.  both >1 -> training/hyperparameters.")

    (out / "patch_psd.json").write_text(json.dumps(
        dict(rows=rows, nseg=int(n), days=int(len(pick)),
             band_10_60=dict(patch=bp / max(bt, 1e-30),
                             multidiff=bm / max(bt, 1e-30))), indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.4, 8.6), sharex=True,
                                 height_ratios=[2, 1], constrained_layout=True)
    st = {"target": ("k", "-", 2.2), "patch sampler": ("#4363d8", "-", 1.8),
          "MultiDiffusion": ("#e6194b", "--", 1.8)}
    for k in names:
        c_, ls, lw = st[k]
        a1.loglog(wl[1:], P[k][1:], color=c_, ls=ls, lw=lw, label=k)
    a1.set_ylabel("residual PSD [cm$^2$ km]")
    a1.legend(frameon=False)
    a1.grid(alpha=0.3, which="both")
    a1.set_title("Does stage 2 reproduce the spectrum of the target it was trained on?\n"
                 f"{n} segments, {len(pick)} test days, identical estimator",
                 fontsize=11, fontweight="bold")
    for k in ("patch sampler", "MultiDiffusion"):
        c_, ls, lw = st[k]
        a2.semilogx(wl[1:], P[k][1:] / np.maximum(P["target"][1:], 1e-30),
                    color=c_, ls=ls, lw=lw, label=k)
    a2.axhline(1.0, color="k", lw=1.0)
    a2.set_xlabel("wavelength [km]")
    a2.set_ylabel("ratio to target")
    a2.grid(alpha=0.3, which="both")
    a2.invert_xaxis()
    fig.savefig(out / "patch_psd.png", dpi=140, bbox_inches="tight")
    print(f"wrote {out}/patch_psd.png")


if __name__ == "__main__":
    main()
