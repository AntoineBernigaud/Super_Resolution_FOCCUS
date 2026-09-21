"""Daily map figures straight from an archive, with no re-sampling.

evaluate.py --only-plots redraws the members by running the sampler again, which needs
a GPU and gives different draws.  When the archive already holds the members -- as it
does for the noise-mode experiments -- plotting from it is both cheaper and shows
exactly the fields the diagnostics were computed on.
"""
import argparse
from pathlib import Path

import numpy as np

import config as C
from plotting import plot_day, plot_day_geo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", type=int, default=0, help="0 = every day archived")
    ap.add_argument("--tag", default="")
    ap.add_argument("--geo", action="store_true", help="also the currents figure")
    ap.add_argument("--prefix", default="day")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s = np.load(C.STATS)
    mean, std = float(s["shared_mean"]), float(s["shared_std"])
    ssha = np.load(C.CACHE_SSHA, mmap_mode="r")
    sla = np.load(C.CACHE_SLA, mmap_mode="r")

    nk = None

    files = sorted(Path(args.archive).glob("pred_*.npz"))
    if args.days:
        files = files[:args.days]
    print(f"{len(files)} days from {args.archive} -> {out}")

    # ONE colour scale for the whole record.  Computed in a first pass over the truth
    # and the members: a per-day scale makes the same colour mean a different sea
    # level in every figure, so days cannot be compared and an animation pulses.
    print("pass 1: colour scale over the whole record", flush=True)
    lo_a, hi_a, sp_a, spd_a = [], [], [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        t = int(z["t"])
        y = np.asarray(ssha[t], np.float32)
        v = np.isfinite(y)
        if v.sum() > 10000:
            q = np.percentile(np.clip(y[v], -C.CLIP_M, C.CLIP_M), [1, 99])
            lo_a.append(q[0]); hi_a.append(q[1])
        e = np.asarray(z["ens"], np.float32) * std + mean
        sp_a.append(np.percentile(e.std(axis=0) * 100.0, 99))
        gy, gx = np.gradient(e.mean(0))
        spd_a.append(np.percentile(np.hypot(gy, gx), 98))
    vmin, vmax = float(np.median(lo_a)), float(np.median(hi_a))
    spread_max = float(np.median(sp_a))
    print(f"  sea level {vmin*100:.1f} .. {vmax*100:.1f} cm, "
          f"spread max {spread_max:.2f} cm")

    # the speed scale is set from the truth on the best-covered day, so it is a
    # physical number rather than one the model's own excess energy inflates
    best = max(files, key=lambda f: float(np.load(f, allow_pickle=True)["cov"]))
    zb = np.load(best, allow_pickle=True)
    yb = np.asarray(ssha[int(zb["t"])], np.float64)
    mb = np.isfinite(yb)
    lat_f = C.LAT0 + (np.arange(C.NLAT_F) + 0.5) * C.DLAT_C / C.REFINE_LAT
    dy = C.DLAT_C / C.REFINE_LAT * 111_320.0
    dx = C.DLON_C / C.REFINE_LON * 111_320.0 * np.cos(np.deg2rad(lat_f))
    fcor = (2 * 7.2921e-5 * np.sin(np.deg2rad(lat_f)))[:, None]
    eta = np.where(mb, np.clip(yb, -C.CLIP_M, C.CLIP_M), np.nan)
    u = -(9.81 / fcor) * np.gradient(eta, axis=0) / dy
    v_ = (9.81 / fcor) * np.gradient(eta, axis=1) / dx[:, None]
    spd = np.hypot(u, v_) * 100.0
    geo_max = float(np.nanpercentile(spd, 98))
    print(f"  geostrophic speed scale 0 .. {geo_max:.1f} cm/s "
          f"(98th pct of the truth on {str(zb['date'])})")
    for f in files:
        z = np.load(f, allow_pickle=True)
        t, date = int(z["t"]), str(z["date"])
        ens = np.asarray(z["ens"], np.float32)
        mu = np.asarray(z["mu"], np.float32)
        y = np.asarray(ssha[t], np.float32)
        m = np.isfinite(y)
        yc = np.where(m, (np.clip(np.nan_to_num(y, nan=0.0), -C.CLIP_M, C.CLIP_M)
                          - mean) / std, np.nan)
        x = np.asarray(sla[t], np.float32)
        mx = np.isfinite(x)
        xc = np.where(mx, (np.nan_to_num(x, nan=0.0) - mean) / std, 0.0)
        extra = None
        if nk is not None:
            nkf = nk.field(t)
            if nkf is None:
                print(f"  {date}: no NorKyst, skipped", flush=True)
                continue
            # normalised units, like every other field plot_day receives
            extra = ("NorKyst", np.where(np.isfinite(nkf) & nk.mask,
                                         (nkf - mean) / std, np.nan).astype(np.float32))
        a = (out, date, xc, ens, yc, mx.astype(np.float32), mean, std)
        p = plot_day(*a, mu=mu, tag=args.tag, vmin=vmin, vmax=vmax,
                     spread_max=spread_max, extra=extra, prefix=args.prefix)
        print(f"  {date} -> {p.name}", flush=True)
        if args.geo:
            pg = plot_day_geo(*a, mu=mu, tag=args.tag, vmax=geo_max,
                              spread_max=geo_max * 0.6, extra=extra,
                              prefix="B" + args.prefix)
            print(f"  {date} -> {pg.name}", flush=True)


if __name__ == "__main__":
    main()
