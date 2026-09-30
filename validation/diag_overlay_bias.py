"""Why the SWOT swath stands out against the reconstruction it is painted on.

A LEVEL offset is visible in the overlay panel of the daily figures -- the swath
sits at a different sea level than the surrounding reconstruction, so its outline
shows as a step.  It is not a texture problem: it is a property of the conditional
mean, so it is measured here without any filtering.

Everything is on the SWOT-observed pixels of the evaluation box, with only the
per-day area mean removed (a constant, not a low-pass) so that "level" and
"structure" can be separated at all:

  bias        mean(pred - truth), a pure level offset, and the map of it
  rmse, std_ratio, corr   the usual moments, for context
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline

import config as C
from data import load_stats


def stats(pred, truth, duacs, w):
    """All moments on the observed pixels, per-day area mean removed."""
    def dm(a):
        return a - np.average(a, weights=w)
    p, t = dm(pred), dm(truth)
    def cov(a, b):
        return np.average(a * b, weights=w)
    return dict(
        bias_cm=100.0 * (np.average(pred, weights=w) - np.average(truth, weights=w)),
        rmse_cm=100.0 * np.sqrt(np.average((pred - truth) ** 2, weights=w)),
        std_ratio=np.sqrt(cov(p, p) / cov(t, t)),
        corr=cov(p, t) / np.sqrt(cov(p, p) * cov(t, t)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="archive_wh13")
    ap.add_argument("--out", default="validation/plots/lam1.0/overlay_bias")
    ap.add_argument("--days", type=int, default=40)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mean, std = load_stats()
    ssha = np.load(C.CACHE_SSHA, mmap_mode="r")
    sla = np.load(C.CACHE_SLA, mmap_mode="r")
    si, sj = slice(128, 768), slice(312, 912)

    lat_f = C.LAT0 + (np.arange(C.NLAT_F) + 0.5) * C.DLAT_C / C.REFINE_LAT
    lon_f = C.LON0 + (np.arange(C.NLON_F) + 0.5) * C.DLON_C / C.REFINE_LON
    lat_c = C.coarse_lat(np.arange(C.NLAT_C))
    lon_c = C.coarse_lon(np.arange(C.NLON_C))

    files = sorted(Path(args.archive).glob("pred_*.npz"))[:args.days]
    print(f"{len(files)} days from {args.archive}\n")

    names = ["DUACS", "mu (deterministic)", "member", "mean of 8", "SWOT truth"]
    rows = {n: [] for n in names}
    nday = 0
    # accumulate a bias map over observed pixels
    bsum = {n: np.zeros((si.stop - si.start, sj.stop - sj.start)) for n in names}
    bcnt = np.zeros_like(bsum["DUACS"])

    for fi, f in enumerate(files):
        z = np.load(f)
        t = int(z["t"])
        truth = np.asarray(ssha[t], np.float64)[si, sj]
        m = np.isfinite(truth)
        if m.sum() < 10_000:
            continue
        ens = np.asarray(z["ens"], np.float64) * std + mean
        mu = np.asarray(z["mu"], np.float64) * std + mean
        c = np.asarray(sla[t], np.float64)
        c = np.where(np.isfinite(c), c, 0.0)
        duacs = RectBivariateSpline(lat_c, lon_c, c)(lat_f, lon_f)[si, sj]

        fields = {"DUACS": duacs, "mu (deterministic)": mu[si, sj],
                  "member": ens[0][si, sj], "mean of 8": ens.mean(0)[si, sj],
                  "SWOT truth": truth}
        w = m.astype(np.float64)
        for n, a in fields.items():
            s = stats(a[m], truth[m], duacs[m], np.ones(int(m.sum())))
            rows[n].append(s)
            bsum[n] += np.where(m, a - truth, 0.0)
        bcnt += m
        nday += 1

        if fi % 10 == 0:
            print(f"  {fi+1}/{len(files)} {str(z['date'])}", flush=True)

    print("\n=== on SWOT-observed pixels, evaluation box, "
          f"{nday} test days ===")
    print(f"{'field':<22}{'bias cm':>9}{'RMSE cm':>9}{'std/truth':>11}{'corr':>7}")
    summ = {}
    for n in names:
        s = {k: float(np.mean([r[k] for r in rows[n]]))
             for k in ("bias_cm", "rmse_cm", "std_ratio", "corr")}
        summ[n] = s
        print(f"{n:<22}{s['bias_cm']:9.3f}{s['rmse_cm']:9.3f}"
              f"{s['std_ratio']:11.3f}{s['corr']:7.3f}")

    (out / "overlay_bias.json").write_text(json.dumps(dict(summary=summ), indent=2))

    # --- bias map -------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = bcnt > 3
    show = ["DUACS", "mu (deterministic)", "member", "mean of 8"]
    fig, axs = plt.subplots(1, len(show), figsize=(5.2 * len(show), 6.2),
                            constrained_layout=True)
    ext = [lon_f[sj][0], lon_f[sj][-1], lat_f[si][0], lat_f[si][-1]]
    asp = 1.0 / np.cos(np.deg2rad(70.0))
    for a, n in zip(axs, show):
        bm = np.where(ok, 100.0 * bsum[n] / np.maximum(bcnt, 1), np.nan)
        im = a.imshow(bm, origin="lower", extent=ext, cmap="RdBu_r",
                      vmin=-4, vmax=4, aspect=asp, interpolation="nearest")
        a.set_title(f"{n} - SWOT   (mean {np.nanmean(bm):+.2f} cm)", fontsize=10)
        a.set_xlabel("longitude")
    axs[0].set_ylabel("latitude")
    fig.colorbar(im, ax=list(axs), fraction=0.02, pad=0.01,
                 label="time-mean error [cm]")
    fig.suptitle("Time-mean error on SWOT-observed pixels "
                 f"({nday} test days)", fontsize=13, fontweight="bold")
    fig.savefig(out / "bias_map.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}/overlay_bias.json, {out}/bias_map.png")


if __name__ == "__main__":
    main()
