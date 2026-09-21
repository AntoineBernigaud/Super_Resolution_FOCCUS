"""Area-mean geostrophic EKE against time, per split.

Reported as an area MEAN, not an area integral.  The SWOT mask changes from day to
day, so an integral over observed pixels would rise and fall with how much of the box
happened to be seen that day and would confound coverage with ocean physics.  All four
fields are evaluated on exactly the same pixels each day -- the SWOT-observed ones
inside the evaluation box -- so the comparison between curves is like for like.

DUACS is cubic-interpolated onto the fine grid rather than block-repeated: this
diagnostic differentiates the field, and a piecewise-constant field puts a delta at
every 8x3 block edge, which dominates its velocity variance (measured: the
block-repeat control sits five orders above smooth DUACS below 20 km).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline

import config as C
from data import load_stats

G, OMEGA = 9.81, 7.2921e-5
FIELDS = ["DUACS", "SWOT truth", "diffusion member", "diffusion mean of 8"]
NORKYST = "NorKyst"
STYLE = {"DUACS": dict(color="#3cb44b", ls="--", lw=1.6),
         "SWOT truth": dict(color="k", ls="-", lw=2.4),
         "diffusion member": dict(color="#e6194b", ls="-", lw=1.6),
         "diffusion mean of 8": dict(color="#911eb4", ls="-", lw=1.6),
         NORKYST: dict(color="#008080", ls="-.", lw=2.0)}


def geo_uv(eta, lat, dy, dx):
    f = (2 * OMEGA * np.sin(np.deg2rad(lat)))[:, None]
    return (-(G / f) * np.gradient(eta, axis=0) / dy,
            (G / f) * np.gradient(eta, axis=1) / dx[:, None])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archives", nargs="+",
                    default=["train=archive_train", "val=archive_val",
                             "test=archive"])
    ap.add_argument("--out", default="runs/eke_timeseries")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mean, std = load_stats()
    ssha = np.load(C.CACHE_SSHA, mmap_mode="r")
    sla = np.load(C.CACHE_SLA, mmap_mode="r")
    si, sj = slice(128, 768), slice(312, 912)          # evaluation box

    lat_f = C.LAT0 + (np.arange(C.NLAT_F) + 0.5) * C.DLAT_C / C.REFINE_LAT
    lon_f = C.LON0 + (np.arange(C.NLON_F) + 0.5) * C.DLON_C / C.REFINE_LON
    lat_c, lon_c = C.coarse_lat(np.arange(C.NLAT_C)), C.coarse_lon(np.arange(C.NLON_C))
    dy = C.DLAT_C / C.REFINE_LAT * 111_320.0
    dx = C.DLON_C / C.REFINE_LON * 111_320.0 * np.cos(np.deg2rad(lat_f))
    la_b, dx_b = lat_f[si], dx[si]

    nk = None

    series = {}
    for spec in args.archives:
        name, d = spec.split("=", 1)
        files = sorted(Path(d).glob("pred_*.npz"))
        if not files:
            print(f"[{name}] no archive at {d}, skipping")
            continue
        rows = []
        for fp in files:
            z = np.load(fp, allow_pickle=True)
            t, date = int(z["t"]), str(z["date"])
            y = np.asarray(ssha[t], np.float32)
            m = np.isfinite(y)[si, sj]
            if m.sum() < 5000:
                continue
            yc = (np.clip(np.nan_to_num(y, nan=0.0), -C.CLIP_M, C.CLIP_M)
                  - mean)[si, sj]
            ens = z["ens"].astype(np.float32) * std
            duacs = RectBivariateSpline(
                lat_c, lon_c,
                np.nan_to_num(np.asarray(sla[t], np.float32) - mean),
                kx=3, ky=3)(lat_f, lon_f)[si, sj]
            F = {"DUACS": duacs, "SWOT truth": yc,
                 "diffusion member": ens[0][si, sj],
                 "diffusion mean of 8": ens.mean(0)[si, sj]}
            if nk is not None:
                nkf = nk.field(t)
                if nkf is None:
                    continue
                nkv = np.isfinite(nkf) & nk.mask
                # restrict EVERY field to the NorKyst footprint, so the curves stay
                # a like-for-like comparison on identical pixels
                m = m & nkv[si, sj]
                F[NORKYST] = np.where(nkv, nkf, 0.0)[si, sj]
            # Gradients need both neighbours observed, so the scoring mask is the
            # interior of the observed region -- identical for every field.
            core = np.zeros_like(m)
            core[1:-1, 1:-1] = (m[2:, 1:-1] & m[:-2, 1:-1]
                                & m[1:-1, 2:] & m[1:-1, :-2] & m[1:-1, 1:-1])
            if core.sum() < 5000:
                continue
            r = {"date": date, "n": int(core.sum()), "coverage": float(m.mean())}
            for k, eta in F.items():
                u, v = geo_uv(eta, la_b, dy, dx_b)
                r[k] = float(1e4 * 0.5 * (u[core] ** 2 + v[core] ** 2).mean())
            rows.append(r)
            print(f"  [{name}] {date}  n={r['n']:>7,}  "
                  + "  ".join(f"{k.split()[0]}={r[k]:.1f}" for k in F), flush=True)
        series[name] = rows

    json.dump(series, open(out / "eke_timeseries.json", "w"), indent=2)
    print("\n=== mean over each split, area-mean geostrophic EKE [cm^2/s^2] ===")
    print(f"  {'split':<8}{'days':>6}" + "".join(f"{f:>22}" for f in FIELDS))
    for name, rows in series.items():
        if not rows:
            continue
        print(f"  {name:<8}{len(rows):>6}"
              + "".join(f"{np.mean([r[f] for r in rows]):>22.1f}" for f in FIELDS))
    _plot(out, series)


def _plot(out, series):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in series if series[n]]
    if not names:
        return
    fig, axs = plt.subplots(1, len(names), figsize=(6.2 * len(names), 4.8),
                            constrained_layout=True, squeeze=False)
    for c, name in enumerate(names):
        rows = series[name]
        dates = np.array([np.datetime64(r["date"]) for r in rows])
        a = axs[0][c]
        for f in FIELDS:
            a.plot(dates, [r[f] for r in rows], marker="o", ms=3,
                   label=f, **STYLE[f])
        a.set_title(f"{name}  ({len(rows)} days)", fontsize=11)
        a.set_xlabel("date")
        a.grid(alpha=0.3)
        a.tick_params(axis="x", rotation=30)
        if c == 0:
            a.set_ylabel("area-mean geostrophic EKE [cm$^2$/s$^2$]")
        if c == len(names) - 1:
            a.legend(fontsize=8)
    fig.suptitle("Area-mean geostrophic EKE inside the evaluation box, "
                 "on SWOT-observed pixels", fontsize=13, fontweight="bold")
    fig.savefig(out / "eke_timeseries.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}/eke_timeseries.png")


if __name__ == "__main__":
    main()
