"""Fractions Skill Score against neighbourhood size, for several thresholds,
averaged over the archived test days.

Follows the structure of FSS_old.py -- percentile thresholds taken from the truth,
FSS over a range of neighbourhood sizes, the 0.5 "useful" line -- with one change
that matters for this dataset.

FSS_old.py sets NaN to 0 before the neighbourhood average, so a window straddling a
gap has its fraction diluted toward zero for BOTH forecast and observation.  With 75%
of the target unobserved and gaps of the size of the nadir stripe, that dilution
dominates at the larger neighbourhoods and drags every method toward the same value.
Here the fractions are normalised by the observed count in each window, i.e. the mean
over the pixels that exist, and windows less than half observed are dropped.  Same
score, computed on the data that is actually there.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

import config as C
from data import load_stats

# 1, then a constant step of 8 out to 200 grid points (~350 km).  n=1 is kept because
# the pointwise value is where the double-penalty signature is clearest.  Watch the
# "fraction actually scored" line: the ">50% observed" window filter was already down
# to 0.71 at n=96, so the far columns rest on fewer windows than the near ones.
NEIGHBOURHOODS = [1] + list(range(8, 201, 8))
PERCENTILES = [50, 75, 90, 95]
STYLES = {
    "DUACS":            dict(color="#3cb44b", linestyle="--", marker="s"),
    "mu (deterministic)": dict(color="#4363d8", linestyle="-.", marker="^"),
    "diffusion member": dict(color="#e6194b", linestyle="-", marker="o"),
    "diffusion mean of 8": dict(color="#f28e8e", linestyle="-", marker="D"),
}


def fss_2d(fc, ob, valid, threshold, sizes):
    """Mask-aware FSS.  Fractions are averaged over observed pixels only."""
    v = valid.astype(np.float64)
    f = ((fc >= threshold) & valid).astype(np.float64)
    o = ((ob >= threshold) & valid).astype(np.float64)
    out = {}
    for n in sizes:
        if n == 1:
            pf, po, w = f, o, v
        else:
            kw = dict(size=n, mode="constant", cval=0.0)
            w = uniform_filter(v, **kw)
            pf = uniform_filter(f, **kw)
            po = uniform_filter(o, **kw)
        use = (w > 0.5) & valid
        if use.sum() == 0:
            out[n] = (np.nan, 0.0)
            continue
        pf, po = pf[use] / w[use], po[use] / w[use]
        num = ((pf - po) ** 2).mean()
        den = (pf ** 2).mean() + (po ** 2).mean()
        # Also return what fraction of the observed pixels actually qualified.  At
        # large neighbourhoods a window can only be half observed near the middle of
        # a swath, so the score is computed on a shrinking, swath-centred subset --
        # the large-n end of the curve is not the same sample as the small-n end.
        frac = float(use.sum() / max(valid.sum(), 1))
        out[n] = (float(1 - num / den) if den > 0 else np.nan, frac)
    return out


# Regions to score over.  "blackbox" is the evaluation box config.py defines and
# plotting.py draws as the dashed rectangle on every daily map -- the default
# fss_vs_scale.png has always been computed on it, the filename just did not say so.
# "area2" is a tighter box away from the northern band where SWOT ground tracks
# converge and the sea ice sits, so it scores the open Nordic Seas only.
REGIONS = {
    "blackbox": (C.EVAL_LAT, C.EVAL_LON),
    "area2": ((66.0, 74.0), (-5.0, 10.0)),
    # the whole grid.  Note this includes the 74-77N band where SWOT ground tracks
    # converge and the sea ice sits -- coverage there is ~4x the rest, so "global"
    # is weighted toward a region the other two boxes deliberately exclude.
    "global": ((C.LAT0, C.LAT0 + C.NLAT_C * C.DLAT_C),
               (C.LON0, C.LON0 + C.NLON_C * C.DLON_C)),
}


def region_slices(lat_rng, lon_rng):
    """Fine-grid slices for a lat/lon box, derived from config -- never hardcoded."""
    lat = C.LAT0 + (np.arange(C.NLAT_F) + 0.5) * C.DLAT_C / C.REFINE_LAT
    lon = C.LON0 + (np.arange(C.NLON_F) + 0.5) * C.DLON_C / C.REFINE_LON
    i = np.nonzero((lat >= lat_rng[0]) & (lat <= lat_rng[1]))[0]
    j = np.nonzero((lon >= lon_rng[0]) & (lon <= lon_rng[1]))[0]
    return slice(i[0], i[-1] + 1), slice(j[0], j[-1] + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="archive_wh13")
    ap.add_argument("--out", default="runs/fss")
    ap.add_argument("--regions", nargs="+", default=["global"],
                    choices=list(REGIONS))
    args = ap.parse_args()

    for name in args.regions:
        lat_rng, lon_rng = REGIONS[name]
        si, sj = region_slices(lat_rng, lon_rng)
        print(f"\n{'='*70}\nregion '{name}': lat {lat_rng[0]}-{lat_rng[1]}, "
              f"lon {lon_rng[0]}-{lon_rng[1]}  -> {si.stop-si.start} x "
              f"{sj.stop-sj.start} px\n{'='*70}")
        run_region(args, name, lat_rng, lon_rng, si, sj)


def run_region(args, region, lat_rng, lon_rng, si, sj):

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mean, std = load_stats()
    files = sorted(Path(args.archive).glob("pred_*.npz"))
    if not files:
        raise SystemExit(f"no archive in {args.archive}/ -- run make_archive.py first")
    print(f"{len(files)} archived days")

    ssha = np.load(C.CACHE_SSHA, mmap_mode="r")
    sla = np.load(C.CACHE_SLA, mmap_mode="r")

    acc = {m: {p: [] for p in PERCENTILES} for m in STYLES}
    FIRST = next(iter(STYLES))
    used = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        t = int(d["t"])
        ens = d["ens"].astype(np.float32)[:, si, sj]
        mu = d["mu"].astype(np.float32)[si, sj]
        y = np.asarray(ssha[t], np.float32)[si, sj]
        y = (np.clip(np.nan_to_num(y, nan=0.0), -C.CLIP_M, C.CLIP_M) - mean) / std
        valid = np.isfinite(np.asarray(ssha[t], np.float32)[si, sj])
        duacs = np.repeat(np.repeat(
            np.asarray(sla[t], np.float32), C.REFINE_LAT, 0), C.REFINE_LON, 1)[si, sj]
        duacs = (np.nan_to_num(duacs, nan=0.0) - mean) / std

        fields = {"DUACS": duacs, "mu (deterministic)": mu,
                  "diffusion member": ens[0], "diffusion mean of 8": ens.mean(0)}
        vals = y[valid]
        if vals.size < 1000:
            continue
        for p in PERCENTILES:
            thr = np.percentile(vals, p)
            for name, fc in fields.items():
                s = fss_2d(fc, y, valid, thr, NEIGHBOURHOODS)
                acc[name][p].append([s[n][0] for n in NEIGHBOURHOODS])
                if name == FIRST:
                    used.append([s[n][1] for n in NEIGHBOURHOODS])
        print(f"  {str(d['date'])}  {100*valid.mean():.1f}% observed", flush=True)

    curves = {m: {p: (np.nanmean(np.array(acc[m][p]), 0),
                      np.nanstd(np.array(acc[m][p]), 0))
                  for p in PERCENTILES} for m in STYLES}
    json.dump({m: {str(p): curves[m][p][0].tolist() for p in PERCENTILES}
               for m in STYLES}, open(out / f"fss_{region}.json", "w"), indent=2)

    u = np.array(used).mean(0)
    print("\n=== fraction of observed pixels actually scored, by neighbourhood ===")
    print("    (a window must be >50% observed to qualify)")
    print("    " + "".join(f"{n:>7}" for n in NEIGHBOURHOODS))
    print("    " + "".join(f"{v:7.3f}" for v in u))

    print("\n=== FSS averaged over the test set (neighbourhood in grid points) ===")
    for p in PERCENTILES:
        print(f"\n  threshold = {p}th percentile of the truth")
        print("    " + "method".ljust(22)
              + "".join(f"{n:>7}" for n in NEIGHBOURHOODS))
        for m in STYLES:
            print("    " + m.ljust(22)
                  + "".join(f"{v:7.3f}" for v in curves[m][p][0]))

    _plot(curves, out, len(files), region, lat_rng, lon_rng)


def _plot(curves, out, nday, region, lat_rng, lon_rng):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    km = [n * 1.739 for n in NEIGHBOURHOODS]
    x = np.arange(len(NEIGHBOURHOODS))
    fig, axes = plt.subplots(1, len(PERCENTILES),
                             figsize=(4.6 * len(PERCENTILES), 4.6),
                             constrained_layout=True, sharey=True)
    for ax, p in zip(np.atleast_1d(axes), PERCENTILES):
        for m, st in STYLES.items():
            mu_, sd_ = curves[m][p]
            ax.plot(x, mu_, label=m, markersize=5, linewidth=1.8, **st)
            ax.fill_between(x, mu_ - sd_, mu_ + sd_, color=st["color"], alpha=0.12)
        ax.axhline(0.5, color="k", lw=1.0, ls="--", alpha=0.6)
        ax.text(x[-1], 0.51, "useful", fontsize=8, va="bottom", ha="right", alpha=0.7)
        step = max(1, len(NEIGHBOURHOODS) // 9)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([f"{n}\n{k:.0f}" for n, k in
                            zip(NEIGHBOURHOODS[::step], km[::step])], fontsize=7)
        ax.set_xlabel("neighbourhood (grid points / km)", fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.set_title(f"threshold: {p}th percentile", fontsize=10, fontweight="bold")
    np.atleast_1d(axes)[0].set_ylabel("FSS", fontsize=10)
    np.atleast_1d(axes)[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Fractions Skill Score, mean over {nday} test days "
                 f"(shading: +-1 sd across days)\n"
                 f"region '{region}': lat {lat_rng[0]:g}-{lat_rng[1]:g}N, "
                 f"lon {lon_rng[0]:g}-{lon_rng[1]:g}E",
                 fontsize=12, fontweight="bold")
    f = out / f"fss_vs_scale_{region}.png"
    fig.savefig(f, dpi=150, bbox_inches="tight")
    print(f"\nwrote {f}")


if __name__ == "__main__":
    main()
