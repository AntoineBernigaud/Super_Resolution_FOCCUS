"""Why the SWOT swath stands out against the reconstruction it is painted on.

Two things are visible in the overlay panel of the daily figures and neither is a
texture problem:

  (1) a LEVEL offset -- the swath sits at a different sea level than the surrounding
      reconstruction, so its outline is visible as a step;
  (2) eddies that DUACS suggests are amplified by the model but are not confirmed by
      the swath.

Both are properties of the conditional mean, so they are measured here without any
filtering: everything is on the SWOT-observed pixels of the evaluation box, with only
the per-day area mean removed (a constant, not a low-pass) so that "level" and
"structure" can be separated at all.

Three questions, three regressions:

  bias        mean(pred - truth).  A pure offset.  Reported per day too, because a
              constant offset over the record is harmless (it is absorbed in the
              anomaly reference) whereas a per-day swing is not.
  slope_D     regression of a field on the DUACS input.  If the model's slope exceeds
              the truth's, the model is amplifying DUACS structure beyond what SWOT
              shows -- observation (2), as a number.
  slope_A     regression of (truth - DUACS) on (pred - DUACS): of the increment the
              model adds to DUACS, how much is real?  1.0 means the additions are
              correctly scaled, <1 means over-amplified, and the correlation says how
              much of it is signal at all.

A control matters here: DUACS sla and SWOT ssha are different products, so part of any
offset may be intrinsic to the data pair rather than made by the model.  truth-vs-DUACS
is therefore reported on the same footing as the model rows.
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
    p, t, d = dm(pred), dm(truth), dm(duacs)
    def cov(a, b):
        return np.average(a * b, weights=w)
    return dict(
        bias_cm=100.0 * (np.average(pred, weights=w) - np.average(truth, weights=w)),
        rmse_cm=100.0 * np.sqrt(np.average((pred - truth) ** 2, weights=w)),
        std_ratio=np.sqrt(cov(p, p) / cov(t, t)),
        corr=cov(p, t) / np.sqrt(cov(p, p) * cov(t, t)),
        slope_D=cov(p, d) / cov(d, d),
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
    daily_bias = []
    # accumulate a bias map over observed pixels
    bsum = {n: np.zeros((si.stop - si.start, sj.stop - sj.start)) for n in names}
    bcnt = np.zeros_like(bsum["DUACS"])
    # increment regression, accumulated over the record
    inc = {n: [0.0, 0.0, 0.0] for n in ["mu (deterministic)", "member", "mean of 8"]}

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
        row = {}
        for n, a in fields.items():
            s = stats(a[m], truth[m], duacs[m], np.ones(int(m.sum())))
            rows[n].append(s)
            row[n] = s["bias_cm"]
            bsum[n] += np.where(m, a - truth, 0.0)
        bcnt += m
        daily_bias.append(dict(date=str(z["date"]), **{k: v for k, v in row.items()}))

        # (truth - DUACS) on (pred - DUACS), pooled over the record
        td = (truth - duacs)[m]
        for n in inc:
            pd = (fields[n] - duacs)[m]
            inc[n][0] += float((pd * td).sum())
            inc[n][1] += float((pd * pd).sum())
            inc[n][2] += float((td * td).sum())

        if fi % 10 == 0:
            print(f"  {fi+1}/{len(files)} {str(z['date'])}", flush=True)

    print("\n=== on SWOT-observed pixels, evaluation box, "
          f"{len(daily_bias)} test days ===")
    print(f"{'field':<22}{'bias cm':>9}{'|bias| spread':>15}{'RMSE cm':>9}"
          f"{'std/truth':>11}{'corr':>7}{'slope on DUACS':>16}")
    summ = {}
    for n in names:
        b = np.array([r["bias_cm"] for r in rows[n]])
        s = {k: float(np.mean([r[k] for r in rows[n]]))
             for k in ("bias_cm", "rmse_cm", "std_ratio", "corr", "slope_D")}
        s["bias_std_cm"] = float(b.std())
        summ[n] = s
        print(f"{n:<22}{s['bias_cm']:9.3f}{s['bias_std_cm']:15.3f}"
              f"{s['rmse_cm']:9.3f}{s['std_ratio']:11.3f}{s['corr']:7.3f}"
              f"{s['slope_D']:16.3f}")

    print("\n=== the increment over DUACS: regress (truth-DUACS) on (pred-DUACS) ===")
    print(f"{'field':<22}{'slope':>8}{'corr':>8}{'|inc| pred/truth':>18}")
    for n, (ct, cp, tt) in inc.items():
        slope = ct / cp
        corr = ct / np.sqrt(cp * tt)
        summ[n]["increment_slope"] = slope
        summ[n]["increment_corr"] = corr
        summ[n]["increment_amp_ratio"] = float(np.sqrt(cp / tt))
        print(f"{n:<22}{slope:8.3f}{corr:8.3f}{np.sqrt(cp/tt):18.3f}")

    (out / "overlay_bias.json").write_text(json.dumps(
        dict(summary=summ, daily=daily_bias), indent=2))

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
                 f"({len(daily_bias)} test days)", fontsize=13, fontweight="bold")
    fig.savefig(out / "bias_map.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}/overlay_bias.json, {out}/bias_map.png")


if __name__ == "__main__":
    main()
