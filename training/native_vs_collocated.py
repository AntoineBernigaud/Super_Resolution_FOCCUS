"""How much does the round trip through the grid cost these metrics?

For the 23 days whose own L3 files we hold, the same swath points can be filled two
ways: from the native L3 ssha_filtered, and from the gridded ssha collocated back onto
them.  Everything else -- the windows, the tapering, the shell ladder -- is identical,
so any difference is the interpolation round trip and nothing else.

This matters because the val and test splits have no native SSH: their truth can only
be the collocated version.  If the two agree over the scales we quote, that comparison
is sound; where they diverge is the floor below which the split results should not be
read.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import config as C
import swath_geom as SG
from data import load_stats
from swath_splits import pack_windows
from swath_transfer import (RATIO, RHO0, G, OMEGA, spacings, wang_window,
                            shells, psd_along, shell_flux)

PAIR = {"native L3": dict(color="k", ls="-", lw=2.4),
        "gridded, collocated back": dict(color="#e6194b", ls="--", lw=1.8)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/native_vs_collocated")
    # so a CANDIDATE target can be scored against native L3 with the identical
    # collocation, windows and shells as the raw gridded product was
    ap.add_argument("--grid-cache", default=None,
                    help="gridded field to collocate (default: the raw cache_ssha)")
    ap.add_argument("--lam-min-km", type=float, default=4.0)
    ap.add_argument("--lam-max-km", type=float, default=512.0)
    ap.add_argument("--min-cols", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mean, std = load_stats()
    ssha_cache = np.load(args.grid_cache or C.CACHE_SSHA, mmap_mode="r")
    idx = np.load(C.PATCH_INDEX)
    dates_all = idx["dates"].astype("datetime64[D]")
    pos = {str(d): i for i, d in enumerate(dates_all)}

    lat_f = C.LAT0 + (np.arange(C.NLAT_F) + 0.5) * C.DLAT_C / C.REFINE_LAT
    lon_f = C.LON0 + (np.arange(C.NLON_F) + 0.5) * C.DLON_C / C.REFINE_LON
    lat0, lat1 = C.LAT0, C.LAT0 + C.NLAT_C * C.DLAT_C
    lon0, lon1 = C.LON0, C.LON0 + C.NLON_C * C.DLON_C

    kfix = shells(args.lam_min_km * 1000.0, args.lam_max_km * 1000.0)
    nb = len(kfix) - 1
    kmid = np.sqrt(kfix[:-1] * kfix[1:]) / (2 * np.pi)
    K1, K2 = np.meshgrid(kmid, kmid, indexing="ij")
    loc = ((K1 / K2) <= RATIO) & ((K1 / K2) >= 1 / RATIO)
    L = int(np.ceil(args.lam_max_km * 1000.0 / 2000.0))

    dates = [d for d in SG.native_dates() if d in pos]
    print(f"{len(dates)} dates with native L3 SSH: {dates[0]} .. {dates[-1]}")

    accP = {k: None for k in PAIR}
    accE = {k: None for k in PAIR}
    accT = {k: np.zeros((nb, nb)) for k in PAIR}
    perPass = {k: [] for k in PAIR}
    nwin = npass = 0

    for date in dates:
        t = pos[date]
        y = np.asarray(ssha_cache[t], np.float32)
        yv = np.isfinite(y)
        yc = np.clip(np.nan_to_num(y, nan=0.0), -C.CLIP_M, C.CLIP_M) - mean

        def rgi(g, method="cubic"):
            return RegularGridInterpolator(
                (lat_f, lon_f), np.nan_to_num(g).astype(np.float64), method=method,
                bounds_error=False, fill_value=np.nan)
        GT, GM = rgi(yc), rgi(yv.astype(np.float64), "linear")

        for _p, fname in SG.native_passes_for(date):
            lat_a, lon_a = SG.geometry(fname)
            nat = SG.ssha(fname)
            inbox = ((lat_a >= lat0) & (lat_a <= lat1)
                     & (lon_a >= lon0) & (lon_a <= lon1))
            if not inbox.any():
                continue
            rows = np.nonzero(inbox.any(axis=1))[0]
            sl0 = slice(rows[0], rows[-1] + 1)
            la_p, lo_p = lat_a[sl0], lon_a[sl0]
            nat_p, ib = nat[sl0], inbox[sl0]
            q = np.stack([la_p.ravel(), lo_p.ravel()], axis=1)
            col = np.where((GM(q).reshape(la_p.shape) > 0.999) & ib,
                           GT(q).reshape(la_p.shape), np.nan)
            # The native field is clipped the same way the gridded one was, so the
            # comparison is not contaminated by the +-0.5 m clip.
            nat_p = np.where(ib, np.clip(nat_p, -C.CLIP_M, C.CLIP_M) - mean, np.nan)

            ok = np.isfinite(nat_p) & np.isfinite(col)     # identical sampling
            wins = pack_windows(ok, L, args.min_cols)
            if not wins:
                continue
            got = {k: [] for k in PAIR}
            for (i0, cols) in wins:
                sl = slice(i0, i0 + L)
                la, lo = la_p[sl][:, cols], lo_p[sl][:, cols]
                ny, nx = la.shape
                d_al, d_ac = spacings(la, lo)
                d_along = float(np.nanmedian(d_al))
                f_cor = 2 * OMEGA * np.sin(np.deg2rad(la))
                win = wang_window(ny)
                fields = {"native L3": nat_p[sl][:, cols],
                          "gridded, collocated back": col[sl][:, cols]}
                if not all(np.isfinite(v).all() for v in fields.values()):
                    continue
                for k, eta in fields.items():
                    u = -(G / f_cor) * np.gradient(eta, axis=0) / d_al
                    v = (G / f_cor) * np.gradient(eta, axis=1) / d_ac
                    T = shell_flux(u, v, d_along, d_ac, kfix, win)
                    accT[k] += T
                    got[k].append(T)
                    p_ = psd_along(eta, d_along, win)
                    e_ = psd_along(u, d_along, win) + psd_along(v, d_along, win)
                    accP[k] = p_ if accP[k] is None else accP[k] + p_
                    accE[k] = e_ if accE[k] is None else accE[k] + e_
                nwin += 1
            if got["native L3"]:
                for k in PAIR:
                    perPass[k].append(np.mean(got[k], axis=0))
                npass += 1
        print(f"  {date}  passes {npass}, windows {nwin}", flush=True)

    if nwin == 0:
        raise SystemExit("no windows")
    for k in PAIR:
        accP[k] /= nwin
        accE[k] /= nwin
        accT[k] /= nwin
    print(f"\n{npass} passes, {nwin} windows")

    n = 2 * (len(accP["native L3"]) - 1)
    freq = np.fft.rfftfreq(n, d=2000.0)
    wl = np.where(freq > 0, 1 / np.maximum(freq, 1e-30), np.inf) / 1000.0

    print("\n=== ratio  collocated / native  (1.0 = the round trip costs nothing) ===")
    probes = [200, 100, 60, 40, 30, 20, 15, 10, 7, 5]
    print("  wavelength [km] " + "".join(f"{p:>8}" for p in probes))
    rows = {}
    for lab, acc in (("SSH PSD", accP), ("EKE PSD", accE)):
        r = acc["gridded, collocated back"] / np.maximum(acc["native L3"], 1e-30)
        vals = [r[np.argmin(np.abs(wl[1:] - p)) + 1] for p in probes]
        rows[lab] = vals
        print(f"  {lab:<15}" + "".join(f"{v:8.3f}" for v in vals))

    def totals(T):
        Tr = T * RHO0
        return (sum(np.nansum(Tr[k:, :k] * loc[k:, :k]) for k in range(1, nb)),
                sum(np.nansum(Tr[k:, :k] * ~loc[k:, :k]) for k in range(1, nb)))

    print("\n=== cross-scale EKE flux (10^-6 W/m^3, +- SE over passes) ===")
    print(f"  {'truth source':<26}{'local':>18}{'nonlocal':>18}{'total':>18}")
    summ = {}
    for k in PAIR:
        a = 1e6 * np.array([totals(T) for T in perPass[k]])
        m_, se = a.mean(0), a.std(0, ddof=1) / np.sqrt(len(a))
        tot = a.sum(1)
        summ[k] = dict(local=[m_[0], se[0]], nonlocal_=[m_[1], se[1]],
                       total=[tot.mean(), tot.std(ddof=1) / np.sqrt(len(a))])
        print(f"  {k:<26}{m_[0]:>11.2f} +-{se[0]:<5.2f}{m_[1]:>11.2f} +-{se[1]:<5.2f}"
              f"{tot.mean():>11.2f} +-{tot.std(ddof=1)/np.sqrt(len(a)):<5.2f}")

    json.dump({"ratio": rows, "probes_km": probes, "flux": summ,
               "npass": npass, "nwin": nwin},
              open(out / "native_vs_collocated.json", "w"), indent=2)
    _plot(out, wl, accP, accE, accT, kmid, loc, nb, npass, nwin)


def _plot(out, wl, P, E, T, kmid, loc, nb, npass, nwin):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    for c, (lab, acc, unit) in enumerate(
            (("SSH", P, "m$^2$/(cycle/m)"), ("geostrophic EKE", E,
                                             "(m/s)$^2$/(cycle/m)"))):
        a = axs[0][c]
        for k, st in PAIR.items():
            a.loglog(wl[1:], acc[k][1:], label=k, **st)
        a.set_xlabel("wavelength [km]"); a.set_ylabel(unit)
        a.set_title(f"{lab} power spectral density"); a.invert_xaxis()
        a.grid(alpha=0.3, which="both"); a.legend(fontsize=8)

    a = axs[0][2]
    for lab, acc, col in (("SSH", P, "#4363d8"), ("EKE", E, "#f58231")):
        r = acc["gridded, collocated back"] / np.maximum(acc["native L3"], 1e-30)
        a.semilogx(wl[1:], r[1:], color=col, lw=1.8, label=lab)
    a.axhline(1.0, color="k", ls="--", lw=1)
    a.set_ylim(0, 2)
    a.set_xlabel("wavelength [km]")
    a.set_ylabel("collocated / native")
    a.set_title("cost of the round trip\n(1.0 = no difference)")
    a.invert_xaxis(); a.grid(alpha=0.3, which="both"); a.legend(fontsize=8)

    x = kmid[1:] * 1000.0
    for c, (idx, t) in enumerate(((0, "Local (within a factor 2)"),
                                  (1, "Nonlocal"), (2, "Total"))):
        a = axs[1][c]
        for k, st in PAIR.items():
            Tr = T[k] * RHO0
            fl = np.array([np.nansum(Tr[i:, :i] * loc[i:, :i]) for i in range(1, nb)])
            fn = np.array([np.nansum(Tr[i:, :i] * ~loc[i:, :i]) for i in range(1, nb)])
            yv = fl if idx == 0 else (fn if idx == 1 else fl + fn)
            a.semilogx(x, 1e6 * yv, label=k, **st)
        a.axhline(0, color="k", lw=0.8, ls="--")
        a.set_xlabel("wavenumber (cpkm)")
        a.set_ylabel("transfer (10$^{-6}$ W/m$^3$)")
        a.set_title(f"cross-scale EKE transfer -- {t}")
        a.grid(alpha=0.3, which="both"); a.legend(fontsize=8)

    fig.suptitle(f"SWOT truth: native L3 swath points vs the gridded product "
                 f"collocated back onto them\n{nwin} windows from {npass} passes, "
                 f"identical sampling for both", fontsize=13, fontweight="bold")
    fig.savefig(out / "native_vs_collocated.png", dpi=135, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}/native_vs_collocated.png")


if __name__ == "__main__":
    main()
