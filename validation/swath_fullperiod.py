"""PSD, EKE PSD and total cross-scale transfer over the WHOLE super-resolved record.

Same swath-geometry method as swath_splits.py, applied to the production dataset
(product/, 1993-2026, members inflated lambda 3.3) instead of the 40-day archives.

WHERE THE WINDOWS GO.  In the SWOT era a window sits wherever SWOT actually observed
on that date.  Before 2023 nothing flew, so every product day borrows the pass
geometry AND the SWOT-observed mask of the record date with the same day of year
(cycling through the ~2.3 record years).  The spatial and seasonal sampling --
including the winter sea-ice gaps SWOT leaves -- is therefore identical to the
SWOT-era figures, and only the fields change.

SWOT itself is pooled over ALL 828 record days, on its own passes and mask.  Model
fields and SWOT therefore have different window counts, and each curve is the
window-weighted mean over its own windows.

Three modes:
    --year YYYY      accumulate the model fields for one year -> sums npz
    --truth          accumulate SWOT over the whole record    -> sums npz
    --combine        pool every npz in --out and plot
"""
import argparse
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_erosion

import config as C
import swath_geom as SG
from data import load_index, load_stats
from swath_splits import pack_windows
from swath_transfer import (STYLE, RHO0, G, OMEGA, spacings, wang_window, shells,
                            psd_along, shell_flux)

MODEL = ["DUACS", "mu (deterministic)", "diffusion member", "diffusion mean of 2",
         "diffusion mean of 4", "diffusion mean of 8"]
TRUTH = "SWOT truth"
LAT_F = C.LAT0 + (np.arange(C.NLAT_F) + 0.5) * C.DLAT_C / C.REFINE_LAT
LON_F = C.LON0 + (np.arange(C.NLON_F) + 0.5) * C.DLON_C / C.REFINE_LON
LAT_C, LON_C = C.coarse_lat(np.arange(C.NLAT_C)), C.coarse_lon(np.arange(C.NLON_C))
BOX = (C.LAT0, C.LAT0 + C.NLAT_C * C.DLAT_C, C.LON0, C.LON0 + C.NLON_C * C.DLON_C)


def rgi(g, la, lo, method="cubic"):
    return RegularGridInterpolator((la, lo), np.nan_to_num(g).astype(np.float64),
                                   method=method, bounds_error=False, fill_value=np.nan)


def accumulate(date, GI, GM, acc, args, kfix, GL=None):
    """Add every window of `date`'s passes where GM (and GL) say valid; mirrors
    swath_splits.analyse exactly from the window loop on."""
    L = int(np.ceil(args.lam_max_km * 1000.0 / 2000.0))
    lat0, lat1, lon0, lon1 = BOX
    for _pass, gfile in SG.passes_for(date):
        lat_a, lon_a = SG.geometry(gfile)
        inbox = (lat_a >= lat0) & (lat_a <= lat1) & (lon_a >= lon0) & (lon_a <= lon1)
        if not inbox.any():
            continue
        rows = np.nonzero(inbox.any(axis=1))[0]
        r0, r1 = rows[0], rows[-1] + 1
        la_p, lo_p, ib = lat_a[r0:r1], lon_a[r0:r1], inbox[r0:r1]
        pts = np.stack([la_p.ravel(), lo_p.ravel()], axis=1)
        ok = (GM(pts).reshape(la_p.shape) > 0.999) & ib
        if GL is not None:
            ok &= GL(pts).reshape(la_p.shape) > 0.999
        for (i0, cols) in pack_windows(ok, L, args.min_cols):
            sl = slice(i0, i0 + L)
            la, lo = la_p[sl][:, cols], lo_p[sl][:, cols]
            ny, nx = la.shape
            d_al, d_ac = spacings(la, lo)
            d_along = float(np.nanmedian(d_al))
            f_cor = 2 * OMEGA * np.sin(np.deg2rad(la))
            q = np.stack([la.ravel(), lo.ravel()], axis=1)
            F = {k: GI[k](q).reshape(ny, nx) for k in GI}
            if not all(np.isfinite(v).all() for v in F.values()):
                continue
            win = wang_window(ny)
            for name, eta in F.items():
                u = -(G / f_cor) * np.gradient(eta, axis=0) / d_al
                v = (G / f_cor) * np.gradient(eta, axis=1) / d_ac
                a = acc.setdefault(name, {"P": 0.0, "E": 0.0, "T": 0.0, "n": 0})
                a["T"] = a["T"] + shell_flux(u, v, d_along, d_ac, kfix, win)
                a["P"] = a["P"] + psd_along(eta, d_along, win)
                a["E"] = a["E"] + psd_along(u, d_along, win) + psd_along(v, d_along, win)
                a["n"] += 1


def save(acc, path, **meta):
    out = {}
    for name, a in acc.items():
        for k in ("P", "E", "T", "n"):
            out[f"{name}|{k}"] = np.asarray(a[k])
    np.savez(path, **out, **{f"meta|{k}": v for k, v in meta.items()})
    print(f"wrote {path}: " + ", ".join(f"{k} {a['n']} windows" for k, a in acc.items()))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--year", type=int)
    g.add_argument("--truth", action="store_true")
    g.add_argument("--combine", action="store_true")
    ap.add_argument("--product", default="product")
    ap.add_argument("--out", default="validation/plots/full_period/swath")
    ap.add_argument("--lam-min-km", type=float, default=4.0)
    ap.add_argument("--lam-max-km", type=float, default=512.0)
    ap.add_argument("--min-cols", type=int, default=8)
    ap.add_argument("--max-days", type=int, default=0, help="testing only")
    args = ap.parse_args()
    out = Path(args.out); (out / "chunks").mkdir(parents=True, exist_ok=True)
    kfix = shells(args.lam_min_km * 1000.0, args.lam_max_km * 1000.0)
    if args.combine:
        return combine(out, kfix)

    mean, _ = load_stats()
    rec = load_index()["dates"].astype("datetime64[D]")
    ssha = np.load(C.CACHE_SSHA, mmap_mode="r")
    acc = {}

    if args.truth:
        for t, d in enumerate(rec):
            y = np.asarray(ssha[t], np.float32)
            yv = np.isfinite(y)
            if yv.sum() < 1000:
                continue
            yc = np.clip(np.nan_to_num(y), -C.CLIP_M, C.CLIP_M) - mean
            accumulate(str(d), {TRUTH: rgi(yc, LAT_F, LON_F)},
                       rgi(yv.astype(float), LAT_F, LON_F, "linear"), acc, args, kfix)
            if t % 100 == 0:
                print(f"  {d}  windows {acc.get(TRUTH, {}).get('n', 0)}", flush=True)
        return save(acc, out / "chunks" / "truth.npz", days=len(rec))

    # geometry donor: the record day with the same day of year, cycling over years
    rdoy = np.array([(d - d.astype("datetime64[Y]")).astype(int) for d in rec])
    files = sorted(Path(args.product, str(args.year)).glob("sr_nordic_sla_*.nc"))
    if args.max_days:
        files = files[:args.max_days]
    ocean_ok = GL = None
    nday = 0
    for f in files:
        with Dataset(f) as ds:
            ds.set_auto_mask(True)
            day = np.datetime64("1950-01-01") + np.timedelta64(int(ds["time"][0]), "D")
            ens = ds["sla"][0].filled(np.nan) - mean
            mu = ds["sla_mu"][0].filled(np.nan) - mean
            du = ds["sla_duacs"][0].filled(np.nan) - mean
        if GL is None:
            # keep the cubic stencil off land: the product is NaN there, which rgi
            # would turn into zeros right at the coast
            ocean_ok = binary_erosion(np.isfinite(mu), iterations=3)
            GL = rgi(ocean_ok.astype(float), LAT_F, LON_F, "linear")
        doy = int((day - day.astype("datetime64[Y]")).astype(int))
        dist = np.minimum(np.abs(rdoy - doy), 365 - np.abs(rdoy - doy))
        cand = np.nonzero(dist == dist.min())[0]
        t = int(cand[int(str(day)[:4]) % len(cand)])
        yv = np.isfinite(np.asarray(ssha[t], np.float32))
        GI = {"DUACS": rgi(du, LAT_C, LON_C),
              "mu (deterministic)": rgi(mu, LAT_F, LON_F),
              "diffusion member": rgi(ens[0], LAT_F, LON_F),
              "diffusion mean of 2": rgi(ens[:2].mean(0), LAT_F, LON_F),
              "diffusion mean of 4": rgi(ens[:4].mean(0), LAT_F, LON_F),
              "diffusion mean of 8": rgi(ens.mean(0), LAT_F, LON_F)}
        accumulate(str(rec[t]), GI, rgi(yv.astype(float), LAT_F, LON_F, "linear"),
                   acc, args, kfix, GL=GL)
        nday += 1
        if nday % 50 == 0:
            print(f"  {day} (geometry of {rec[t]})  windows "
                  f"{acc.get(MODEL[0], {}).get('n', 0)}", flush=True)
    save(acc, out / "chunks" / f"model_{args.year}.npz", days=nday)


def combine(out, kfix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tot, ndays = {}, 0
    chunks = sorted((out / "chunks").glob("*.npz"))
    for c in chunks:
        z = np.load(c)
        if c.name.startswith("model_"):
            ndays += int(z["meta|days"])
        for k in z.files:
            if k.startswith("meta|"):
                continue
            name, q = k.split("|")
            a = tot.setdefault(name, {"P": 0.0, "E": 0.0, "T": 0.0, "n": 0})
            a[q] = a[q] + z[k]
    names = [f for f in MODEL + [TRUTH] if f in tot]
    missing = [y for y in range(1993, 2027)
               if not (out / "chunks" / f"model_{y}.npz").exists()]
    if missing:
        print(f"WARNING: no chunk for years {missing}")
    nm = int(tot[MODEL[0]]["n"]); nt = int(tot.get(TRUTH, {"n": 0})["n"])
    tag = (f"model fields: {ndays:,} days, {nm:,} windows   |   "
           f"SWOT: its whole record, {nt:,} windows")
    print(tag)

    nb = len(kfix) - 1
    kmid = np.sqrt(kfix[:-1] * kfix[1:]) / (2 * np.pi)
    lam = 1.0 / (kmid[1:] * 1000.0)
    n_fft = 2 * (len(tot[MODEL[0]]["P"]) - 1)
    freq = np.fft.rfftfreq(n_fft, d=2000.0)
    wl = np.where(freq > 0, 1 / np.maximum(freq, 1e-30), np.inf) / 1000.0

    for key, ylab, fname in (("P", "SSH PSD [m$^2$/cpm]", "psd_full_period"),
                             ("E", "geostrophic EKE PSD [m$^3$/s$^2$]",
                              "eke_psd_full_period")):
        fig, a = plt.subplots(figsize=(8.6, 6.4), constrained_layout=True)
        for f_ in names:
            a.loglog(wl[1:], (tot[f_][key] / tot[f_]["n"])[1:], label=f_, **STYLE[f_])
        a.set_xlabel("wavelength [km]"); a.set_ylabel(ylab); a.invert_xaxis()
        a.grid(alpha=0.3, which="both"); a.legend(frameon=False, fontsize=8)
        a.set_title(f"{'SSH' if key == 'P' else 'EKE'} spectrum, FULL PERIOD 1993-2026\n"
                    + tag, fontsize=10, fontweight="bold")
        fig.savefig(out / f"{fname}.png", dpi=135, bbox_inches="tight"); plt.close(fig)
        print(f"wrote {out}/{fname}.png")

    fig, a = plt.subplots(figsize=(8.6, 6.4), constrained_layout=True)
    hi = 0.0
    for f_ in names:
        T = tot[f_]["T"] / tot[f_]["n"] * RHO0
        tt = np.array([np.nansum(T[k:, :k]) for k in range(1, nb)]) * 1e6
        a.plot(lam, tt, label=f_, **STYLE[f_])
        if f_ in (TRUTH, "DUACS", "mu (deterministic)"):
            hi = max(hi, float(np.nanmax(np.abs(tt))))
    a.set_xscale("log"); a.set_yscale("symlog", linthresh=max(hi, 1e-3))
    a.axhline(0.0, color="k", lw=0.8); a.invert_xaxis()
    a.set_xlabel("wavelength [km]"); a.set_ylabel("total transfer (10$^{-6}$ W/m$^3$)")
    a.grid(alpha=0.3, which="both"); a.legend(frameon=False, fontsize=8)
    a.set_title("Total cross-scale EKE transfer, FULL PERIOD 1993-2026\n" + tag,
                fontsize=10, fontweight="bold")
    fig.savefig(out / "cross_scale_transfer_full_period.png", dpi=135, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}/cross_scale_transfer_full_period.png")


if __name__ == "__main__":
    main()
