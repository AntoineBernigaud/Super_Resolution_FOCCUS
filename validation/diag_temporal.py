"""Is the super-resolution consistent from one day to the next?

Every day is sampled independently: stage 2 draws fresh noise, so the generated
texture is redrawn from scratch each time while mu evolves smoothly with DUACS.  The
ocean does not do that -- the mesoscale decorrelates over 10-20 days, so consecutive
days should look almost identical.  Nothing in the training or sampling enforces this,
and a reanalysis product with texture that flickers daily would carry spurious
high-frequency variability and unphysical accelerations.

Two measurements, both on pixels observed on BOTH days of a pair so that model and
truth are compared on identical samples:

  lagged correlation   corr(f(t), f(t+lag)).  Truth sets the bar.  A field whose
                       variance is partly independent noise decorrelates faster.
  1-day difference     std(f(t+1) - f(t)).  If the texture is independent between
                       days it contributes sqrt(2) x its own std to this, so the
                       inflation over truth measures the flicker directly.

Needs CONSECUTIVE days: the standard archives are spaced ~3 days apart by
make_archive's even sampling, which is right for time-mean statistics and useless
here.  Build one with `make_archive.py --dates ...`.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import config as C
from data import load_stats
from fss_analysis import REGIONS, region_slices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="archive_consecutive")
    ap.add_argument("--out", default="validation/plots/lam1.0/temporal")
    ap.add_argument("--region", default="blackbox", choices=list(REGIONS))
    ap.add_argument("--max-lag", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mean, std = load_stats()
    ssha = np.load(C.CACHE_SSHA, mmap_mode="r")
    sla = np.load(C.CACHE_SLA, mmap_mode="r")
    si, sj = region_slices(*REGIONS[args.region])

    files = sorted(Path(args.archive).glob("pred_*.npz"))
    if len(files) < 2:
        raise SystemExit(f"need >=2 days in {args.archive}")
    days = {}
    for f in files:
        d = np.load(f, allow_pickle=True)
        t = int(d["t"])
        y = np.asarray(ssha[t], np.float64)[si, sj]
        m = np.isfinite(y)
        ens = (np.asarray(d["ens"], np.float64)[:, si, sj] * std + mean) * 100.0
        duacs = np.repeat(np.repeat(np.asarray(sla[t], np.float64),
                                    C.REFINE_LAT, 0), C.REFINE_LON, 1)[si, sj] * 100.0
        days[t] = dict(
            date=str(d["date"]), m=m,
            truth=np.clip(y, -C.CLIP_M, C.CLIP_M) * 100.0,
            duacs=np.nan_to_num(duacs),
            mu=(np.asarray(d["mu"], np.float64)[si, sj] * std + mean) * 100.0,
            member=ens[0], mean8=ens.mean(0))
    ts = sorted(days)
    print(f"{len(ts)} days: {days[ts[0]]['date']} .. {days[ts[-1]]['date']}")

    names = ["truth", "duacs", "mu", "member", "mean8"]
    res = {"region": args.region, "lags": {}}
    print(f"\n=== lagged correlation, region '{args.region}' "
          f"(pixels observed on BOTH days) ===")
    print(f"{'lag d':>6}{'pairs':>7}{'pixels':>11}" + "".join(f"{n:>10}" for n in names))
    for lag in range(1, args.max_lag + 1):
        num = {n: 0.0 for n in names}
        den1 = {n: 0.0 for n in names}
        den2 = {n: 0.0 for n in names}
        npair = npx = 0
        for t in ts:
            u = t + lag
            if u not in days:
                continue
            a, b = days[t], days[u]
            both = a["m"] & b["m"]
            if both.sum() < 500:
                continue
            for n in names:
                x, yv = a[n][both], b[n][both]
                x = x - x.mean(); yv = yv - yv.mean()
                num[n] += float((x * yv).sum())
                den1[n] += float((x * x).sum())
                den2[n] += float((yv * yv).sum())
            npair += 1
            npx += int(both.sum())
        if npair == 0:
            continue
        row = {n: num[n] / max(np.sqrt(den1[n] * den2[n]), 1e-30) for n in names}
        res["lags"][lag] = dict(pairs=npair, pixels=npx, **row)
        print(f"{lag:6d}{npair:7d}{npx:11,}" + "".join(f"{row[n]:10.3f}" for n in names))

    # --- day-to-day difference ---------------------------------------------------
    print(f"\n=== std of the 1-day difference [cm] (same pixels) ===")
    acc = {n: [0.0, 0] for n in names}
    for t in ts:
        u = t + 1
        if u not in days:
            continue
        both = days[t]["m"] & days[u]["m"]
        if both.sum() < 500:
            continue
        for n in names:
            df = days[u][n][both] - days[t][n][both]
            acc[n][0] += float(((df - df.mean()) ** 2).sum())
            acc[n][1] += int(both.sum())
    diff = {n: float(np.sqrt(acc[n][0] / max(acc[n][1], 1))) for n in names}
    res["day_diff_std_cm"] = diff
    for n in names:
        extra = (f"   {diff[n]/max(diff['truth'],1e-9):5.2f}x truth"
                 if n != "truth" else "")
        print(f"  {n:<10}{diff[n]:8.3f} cm{extra}")
    (out / "temporal.json").write_text(json.dumps(res, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    st = {"truth": dict(color="k", lw=2.6), "duacs": dict(color="#3cb44b", ls="--"),
          "mu": dict(color="#4363d8"), "member": dict(color="#e6194b"),
          "mean8": dict(color="#911eb4")}
    lags = sorted(res["lags"])
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    for n in names:
        axs[0].plot(lags, [res["lags"][l][n] for l in lags], marker="o",
                    label=n, **st[n])
    axs[0].set_xlabel("lag [days]"); axs[0].set_ylabel("correlation")
    axs[0].grid(alpha=0.3); axs[0].legend(fontsize=9)
    axs[0].set_title("Lagged correlation -- a field carrying independent daily\n"
                     "noise decorrelates faster than the ocean does", fontsize=10,
                     fontweight="bold")
    b = axs[1].bar(range(len(names)), [diff[n] for n in names],
                   color=[st[n].get("color") for n in names])
    axs[1].bar_label(b, fmt="%.2f", fontsize=9)
    axs[1].set_xticks(range(len(names))); axs[1].set_xticklabels(names)
    axs[1].set_ylabel("std of f(t+1) - f(t)  [cm]")
    axs[1].grid(axis="y", alpha=0.3)
    axs[1].set_title("Day-to-day change\nexcess over truth = the flicker",
                     fontsize=10, fontweight="bold")
    fig.suptitle(f"Temporal consistency, region '{args.region}', "
                 f"{len(ts)} consecutive days", fontsize=12, fontweight="bold")
    fig.savefig(out / "temporal_consistency.png", dpi=140, bbox_inches="tight")
    print(f"\nwrote {out}/temporal_consistency.png, {out}/temporal.json")


if __name__ == "__main__":
    main()
