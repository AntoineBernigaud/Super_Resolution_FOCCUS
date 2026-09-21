"""Where does the model improve on DUACS, and where does it make things worse?

Two products, both over the archived test days and always on SWOT-observed pixels:

  maps    per-pixel RMSE for each field, plus DIFFERENCE maps against the two
          references that matter -- DUACS (does super-resolving help at all?) and mu
          (does the generative stage help on top of stage 1?).  Blue = the model is
          better there, red = worse.  A per-pixel RMSE over ~40 days is noisy where
          coverage is thin, so pixels seen fewer than --min-days times are left blank
          rather than drawn as if they meant something.
  table   RMSE per region for three individual members (so member-to-member spread is
          visible rather than hidden behind an average), the mean of 8, mu and DUACS.
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
    ap.add_argument("--archive", default="archive_wh13")
    ap.add_argument("--out", default="validation/plots/lam1.0/rmse")
    ap.add_argument("--min-days", type=int, default=6)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mean, std = load_stats()
    ssha = np.load(C.CACHE_SSHA, mmap_mode="r")
    sla = np.load(C.CACHE_SLA, mmap_mode="r")

    names = ["DUACS", "mu", "member 1", "member 2", "member 3", "mean of 8"]
    nk = None
    files = sorted(Path(args.archive).glob("pred_*.npz"))
    if not files:
        raise SystemExit(f"no archive in {args.archive}")
    print(f"{len(files)} archived days from {args.archive}")

    se = {n: np.zeros((C.NLAT_F, C.NLON_F)) for n in names}
    cnt = np.zeros((C.NLAT_F, C.NLON_F))
    for k, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        t = int(d["t"])
        y = np.asarray(ssha[t], np.float64)
        m = np.isfinite(y)
        if nk is not None:
            nkf = nk.field(t)
            if nkf is None:
                continue
            m = m & np.isfinite(nkf) & nk.mask
        if m.sum() < 1000:
            continue
        yc = np.clip(y, -C.CLIP_M, C.CLIP_M) * 100.0            # cm
        ens = (np.asarray(d["ens"], np.float64) * std + mean) * 100.0
        mu = (np.asarray(d["mu"], np.float64) * std + mean) * 100.0
        duacs = np.repeat(np.repeat(np.asarray(sla[t], np.float64),
                                    C.REFINE_LAT, 0), C.REFINE_LON, 1) * 100.0
        fields = {"DUACS": np.nan_to_num(duacs), "mu": mu, "member 1": ens[0],
                  "member 2": ens[1], "member 3": ens[2], "mean of 8": ens.mean(0)}
        if nk is not None:
            # NorKyst is an anomaly about its own 120-day mean and so has ~zero mean,
            # while ssha / sla / mu all carry the record mean (+11.3 cm).  Comparing
            # them directly would make NorKyst's RMSE the offset and nothing else.
            # Remove each field's mean over the pixels actually scored, on this day --
            # INCLUDING the truth, or the offset simply moves into the residual.
            # These RMSEs are therefore pattern errors and are NOT comparable to the
            # unrestricted table, which is not demeaned.
            fields["NorKyst"] = nkf * 100.0
            yc = yc - np.mean(yc[m])
            for n in list(fields):
                fields[n] = fields[n] - np.mean(fields[n][m])
        for n, a in fields.items():
            se[n] += np.where(m, (a - yc) ** 2, 0.0)
        cnt += m
        if k % 10 == 0:
            print(f"  {k + 1}/{len(files)} {str(d['date'])}", flush=True)

    ok = cnt >= args.min_days
    rmse = {n: np.where(ok, np.sqrt(se[n] / np.maximum(cnt, 1)), np.nan)
            for n in names}

    # --- table -----------------------------------------------------------------
    tab = {}
    print(f"\n=== RMSE [cm] on SWOT-observed pixels, {len(files)} test days ===")
    hdr = f"{'region':<10}" + "".join(f"{n:>12}" for n in names)
    print(hdr)
    for region in REGIONS:
        si, sj = region_slices(*REGIONS[region])
        row = {}
        for n in names:
            s = se[n][si, sj].sum()
            c = cnt[si, sj].sum()
            row[n] = float(np.sqrt(s / max(c, 1)))
        tab[region] = row
        print(f"{region:<10}" + "".join(f"{row[n]:12.3f}" for n in names))
    (out / "rmse_table.json").write_text(json.dumps(tab, indent=2))

    # --- maps ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lat = C.LAT0 + (np.arange(C.NLAT_F) + 0.5) * C.DLAT_C / C.REFINE_LAT
    lon = C.LON0 + (np.arange(C.NLON_F) + 0.5) * C.DLON_C / C.REFINE_LON
    ext = [lon[0], lon[-1], lat[0], lat[-1]]
    asp = 1.0 / np.cos(np.deg2rad(70.0))

    show = (["NorKyst", "DUACS", "mu", "member 1", "mean of 8"] if "NorKyst" in names
            else ["DUACS", "mu", "member 1", "mean of 8"])
    fig, axs = plt.subplots(1, len(show), figsize=(5.1 * len(show), 6.4),
                            constrained_layout=True)
    hi = np.nanpercentile(rmse["member 1"], 98)
    for a, n in zip(axs, show):
        im = a.imshow(rmse[n], origin="lower", extent=ext, cmap="viridis",
                      vmin=0, vmax=hi, aspect=asp, interpolation="nearest")
        a.set_title(f"{n}   (median {np.nanmedian(rmse[n]):.2f} cm)", fontsize=10)
        a.set_xlabel("longitude")
    axs[0].set_ylabel("latitude")
    fig.colorbar(im, ax=list(axs), fraction=0.02, pad=0.01, label="RMSE [cm]")
    fig.suptitle(f"Per-pixel RMSE against SWOT, {len(files)} test days "
                 f"(pixels seen < {args.min_days} days left blank)",
                 fontsize=13, fontweight="bold")
    fig.savefig(out / "rmse_map.png", dpi=135, bbox_inches="tight")
    plt.close(fig)

    pairs = [("mu", "DUACS", "does stage 1 beat the input?"),
             ("member 1", "DUACS", "does one member beat the input?"),
             ("mean of 8", "DUACS", "does the ensemble mean beat the input?"),
             ("mean of 8", "mu", "does the ensemble mean beat stage 1?")]
    if "NorKyst" in names:
        pairs.append(("NorKyst", "DUACS", "does NorKyst beat the input?"))
    fig, axs = plt.subplots(1, len(pairs), figsize=(4.7 * len(pairs), 6.4),
                            constrained_layout=True)
    for a, (x, ref, q) in zip(np.atleast_1d(axs), pairs):
        dmap = rmse[x] - rmse[ref]
        v = np.nanpercentile(np.abs(dmap), 98)
        im = a.imshow(dmap, origin="lower", extent=ext, cmap="RdBu_r",
                      vmin=-v, vmax=v, aspect=asp, interpolation="nearest")
        frac = np.nanmean(dmap < 0)
        a.set_title(f"{x} - {ref}\n{q}\nbetter on {100*frac:.0f}% of pixels",
                    fontsize=9)
        a.set_xlabel("longitude")
        fig.colorbar(im, ax=a, fraction=0.04, pad=0.01,
                     label="RMSE difference [cm]   (blue = better)")
    axs[0].set_ylabel("latitude")
    fig.suptitle("Where the model helps and where it hurts", fontsize=13,
                 fontweight="bold")
    fig.savefig(out / "rmse_difference_map.png", dpi=135, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}/rmse_map.png, {out}/rmse_difference_map.png, "
          f"{out}/rmse_table.json")


if __name__ == "__main__":
    main()
