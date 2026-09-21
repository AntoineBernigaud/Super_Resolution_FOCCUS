"""Score the ensemble with a metric that is fair to an ensemble.

FSS and RMSE are DETERMINISTIC scores applied to a probabilistic product.  A single
member puts full-amplitude structure where the truth does not have it, so it takes a
false alarm and a miss for the same feature -- penalised twice for having the right
variance, while a smooth field is never penalised for making no small-scale claim.
That is the double penalty, and it guarantees a member loses at small neighbourhoods
however good the model gets.  It says nothing about whether the ensemble is useful.

CRPS is the proper score for this.  For a deterministic forecast it reduces exactly to
|forecast - truth|, so mu, DUACS and the ensemble mean are directly comparable to the
ensemble's CRPS on the same pixels, with no handicap either way:

    CRPS(ensemble)  vs  MAE(mu)  vs  MAE(DUACS)  vs  MAE(mean of 8)

The fair (unbiased) estimator from metrics.py is used -- with m=8 the biased form
would flatter the ensemble by ~7%.

The rank histogram is reported alongside: flat = calibrated, U-shaped = under-
dispersed (truth falls outside the ensemble too often), dome = over-dispersed.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import config as C
from data import load_stats
from metrics import crps_ensemble, rank_histogram
from fss_analysis import REGIONS, region_slices


def diagnose_ranks(rh):
    """Separate BIAS from DISPERSION in a rank histogram.

    These are different shapes and the first version of this conflated them: it tested
    p[0] + p[-1] against a flat baseline, which a purely one-sided ramp also trips.
    A monotone ramp is bias -- the truth sits near one end of the ensemble too often --
    while a symmetric U is under-dispersion.  The two need separate statistics:

      mean rank      sum(i * p_i) / m.  0.5 when unbiased; below 0.5 means the truth
                     tends to the bottom of the ensemble, i.e. the members run HIGH.
      ends ratio     computed on the SYMMETRISED histogram, so any ramp cancels before
                     the tails are measured.  >1 under-dispersed, <1 over-dispersed.
    """
    m = len(rh) - 1
    i = np.arange(len(rh))
    mean_rank = float((i * rh).sum() / max(m, 1))
    sym = 0.5 * (rh + rh[::-1])
    ends_ratio = float((sym[0] + sym[-1]) / (2.0 / len(rh)))
    bits = []
    if mean_rank < 0.45:
        bits.append("members biased HIGH")
    elif mean_rank > 0.55:
        bits.append("members biased LOW")
    if ends_ratio > 1.25:
        bits.append("under-dispersed")
    elif ends_ratio < 0.80:
        bits.append("over-dispersed")
    return mean_rank, ends_ratio, ", ".join(bits) or "calibrated"


def _plots(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tags = list(res)
    if not tags:
        return
    regions = list(res[tags[0]])
    methods = [("crps", "8-member ensemble (CRPS)", "#e6194b"),
               ("mu", "mu (MAE)", "#4363d8"),
               ("duacs", "DUACS (MAE)", "#3cb44b"),
               ("ens_mean", "mean of 8 (MAE)", "#911eb4"),
               ("member", "single member (MAE)", "#f58231")]
    # NorKyst is scored in the table but was missing from the figure: this list was
    # hardcoded.  Only add it when it is actually present in the results.
    if any("norkyst" in res[t][r] for t in tags for r in res[t]):
        methods.append(("norkyst", "NorKyst (MAE)", "#008080"))
    fig, axs = plt.subplots(2, len(tags), figsize=(7.4 * len(tags), 9.4),
                            constrained_layout=True, squeeze=False)
    for c, tag in enumerate(tags):
        ax = axs[0][c]
        w = 0.15
        x = np.arange(len(regions))
        w = 0.8 / len(methods)
        for k, (key, lab, col) in enumerate(methods):
            v = [res[tag][r].get(key, np.nan) for r in regions]
            b = ax.bar(x + (k - (len(methods) - 1) / 2) * w, v, w,
                       label=lab, color=col)
            ax.bar_label(b, fmt="%.2f", fontsize=7, padding=1)
        ax.set_xticks(x)
        ax.set_xticklabels(regions)
        ax.set_ylabel("score [cm]   (lower is better)")
        ax.set_title(f"{tag}: CRPS of the ensemble vs MAE of each "
                     f"deterministic field\n"
                     f"(CRPS reduces to MAE for a point forecast, so these "
                     f"are directly comparable)",
                     fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        if c == 0:
            ax.legend(fontsize=8, ncol=2)

        ax = axs[1][c]
        for r in regions:
            rh = np.array(res[tag][r]["rank_histogram"])
            ax.plot(np.arange(len(rh)), rh, marker="o", label=r, lw=1.8)
        ax.axhline(1.0 / len(rh), color="k", ls="--", lw=1.0,
                   label="flat = calibrated")
        ax.set_xlabel("rank of the truth within the 8-member ensemble")
        ax.set_ylabel("frequency")
        ax.set_ylim(0, None)
        ax.grid(alpha=0.3)
        ax.set_title("Rank histogram\nramp = bias, symmetric U = under-dispersion",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
    fig.savefig(out / "crps_summary.png", dpi=145, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}/crps_summary.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archives", nargs="+",
                    default=["lowpass=archive_lowpass_target",
                             "reference=archive"])
    ap.add_argument("--regions", nargs="+", default=["global"],
                    choices=list(REGIONS))
    ap.add_argument("--out", default="validation/plots/lam1.0/crps")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mean, std = load_stats()
    ssha = np.load(C.CACHE_SSHA, mmap_mode="r")
    sla = np.load(C.CACHE_SLA, mmap_mode="r")
    nk = None
    res = {}

    for spec in args.archives:
        tag, arch = spec.split("=", 1)
        files = sorted(Path(arch).glob("pred_*.npz"))
        if not files:
            print(f"{arch}: nothing archived, skipping")
            continue
        res[tag] = {}
        for region in args.regions:
            lat_rng, lon_rng = REGIONS[region]
            si, sj = region_slices(lat_rng, lon_rng)
            keys = ["crps", "mu", "duacs", "ens_mean", "member"]
            if nk is not None:
                keys.append("norkyst")
            acc = {k: 0.0 for k in keys}
            n = 0
            ranks = np.zeros(9)
            for f in files:
                d = np.load(f, allow_pickle=True)
                t = int(d["t"])
                y = np.asarray(ssha[t], np.float32)[si, sj]
                v = np.isfinite(y)
                nkf = None
                if nk is not None:
                    nkf = nk.field(t)
                    if nkf is None:
                        continue
                    nkf = nkf[si, sj]
                    v = v & np.isfinite(nkf) & nk.mask[si, sj]
                if v.sum() < 1000:
                    continue
                # everything in cm, on the observed pixels of this region
                o = (np.clip(y[v], -C.CLIP_M, C.CLIP_M)) * 100.0
                ens = (d["ens"].astype(np.float32)[:, si, sj][:, v]
                       * std + mean) * 100.0
                mu = (d["mu"].astype(np.float32)[si, sj][v] * std + mean) * 100.0
                duacs = np.repeat(np.repeat(np.asarray(sla[t], np.float32),
                                            C.REFINE_LAT, 0),
                                  C.REFINE_LON, 1)[si, sj][v] * 100.0
                if nk is not None:
                    nkv = nkf[v] * 100.0
                    dm = lambda a: a - a.mean(axis=-1, keepdims=True)
                    o, ens, mu, duacs = (dm(o), dm(ens), dm(mu),
                                         dm(np.nan_to_num(duacs)))
                    nkv = nkv - nkv.mean()
                    acc["norkyst"] += float(np.abs(nkv - o).sum())
                acc["crps"] += float(crps_ensemble(ens, o).sum())
                acc["mu"] += float(np.abs(mu - o).sum())
                acc["duacs"] += float(np.abs(np.nan_to_num(duacs) - o).sum())
                acc["ens_mean"] += float(np.abs(ens.mean(0) - o).sum())
                acc["member"] += float(np.abs(ens[0] - o).sum())
                ranks += rank_histogram(ens, o)
                n += int(v.sum())
            if n == 0:
                continue
            r = {k: acc[k] / n for k in acc}
            r["n"] = n
            r["rank_histogram"] = (ranks / max(ranks.sum(), 1)).tolist()
            res[tag][region] = r
            print(f"\n=== {tag}, region '{region}'  ({n:,} pixels) ===")
            print(f"  CRPS, 8-member ensemble (fair)   {r['crps']:7.3f} cm")
            print(f"  MAE,  mu (deterministic)         {r['mu']:7.3f} cm")
            print(f"  MAE,  DUACS                      {r['duacs']:7.3f} cm")
            print(f"  MAE,  mean of 8                  {r['ens_mean']:7.3f} cm")
            print(f"  MAE,  single member              {r['member']:7.3f} cm")
            if "norkyst" in r:
                print(f"  MAE,  NorKyst (free-running model){r['norkyst']:7.3f} cm")
            cands = [("ensemble", r["crps"]), ("mu", r["mu"]),
                     ("DUACS", r["duacs"]), ("mean of 8", r["ens_mean"])]
            if "norkyst" in r:
                cands.append(("NorKyst", r["norkyst"]))
            best = min(cands, key=lambda kv: kv[1])
            print(f"  -> best: {best[0]} ({best[1]:.3f} cm)")
            rh = np.array(r["rank_histogram"])
            bias, disp, label = diagnose_ranks(rh)
            r["rank_mean"], r["rank_ends_ratio"] = bias, disp
            print(f"  rank histogram ({len(rh)} bins): "
                  + " ".join(f"{x:.3f}" for x in rh))
            print(f"    mean rank {bias:.3f} (0.5 = unbiased), "
                  f"ends ratio {disp:.2f} (1 = calibrated)  -> {label}")

    (out / "crps.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}/crps.json")
    _plots(res, out)


if __name__ == "__main__":
    main()
