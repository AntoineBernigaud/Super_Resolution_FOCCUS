"""PSD and cross-scale EKE transfer in SWOT's own swath geometry.

This replaces the gridded version in eke_transfer.py, which failed for three reasons
that all trace back to one cause -- the regridded product has no long, fully-sampled,
low-noise transects:

  * no fully-observed rectangle larger than 64x32 exists, capping the analysis at
    110 km against Wang et al's 10-1500 km;
  * 100% of those tiles fell north of 74N, outside the evaluation box;
  * every estimate had a standard error larger than itself.

In swath coordinates a pass IS a rectangle, which is why their method works there.
Measured on these files: a pass crosses our box over a median of 858 along-track
lines, up to 1075 -- roughly 2150 km, so the shell ladder spans ~4 km to ~1000 km.

Convention, following their script: x = across-track, y = along-track,
u = -(g/f) d(eta)/dy, v = +(g/f) d(eta)/dx, and the scale decomposition is a sharp
low-pass in the ALONG-TRACK wavenumber only.  Files are kept in their native
(num_lines, num_pixels) order, so along-track is axis 0 here where it is axis 1 in
theirs.  Distx/Disty come from haversine between neighbouring pixels -- this is
exactly what their SWOT_dist_angle<N>.mat holds, and the only reason it has to be
precomputed for them is that swath coordinates are curved.

Fill values: read through netCDF4's _FillValue masking.  The `< -1e8` rule works only
on the raw ints -- ssha_filtered is int32 with a scale factor and its fill lands at
-214748.365 after scaling, which that threshold silently lets through.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator

import config as C
from data import load_stats

G, OMEGA, RHO0 = 9.81, 7.2921e-5, 1025.0
RATIO, SCA = 2.0, 1.2
RE = 6371000.0
FIELDS = ["DUACS", "DUACS block-repeat", "mu (deterministic)",
          "diffusion member", "diffusion mean of 2", "diffusion mean of 4",
          "diffusion mean of 8", "SWOT truth"]
# NorKyst joins the SAME swath-geometry pipeline rather than getting its own
# along-latitude spectra: comparability requires one method for every field, and the
# only method that works for SWOT -- which has gaps -- is along-track in native swath
# coordinates.  enable_norkyst() appends it so the default field list is unchanged
# for every run that has no NorKyst data.
COLORS = {"DUACS": "#3cb44b", "DUACS block-repeat": "0.55",
          "mu (deterministic)": "#4363d8", "diffusion member": "#e6194b",
          "diffusion mean of 2": "#f58231", "diffusion mean of 4": "#f28e8e",
          "diffusion mean of 8": "#911eb4", "SWOT truth": "k"}
COLORS["NorKyst"] = "#008080"


def _mkstyle():
    return {f: dict(color=COLORS[f],
                    lw=2.4 if f in ("SWOT truth", "NorKyst") else 1.6,
                    ls="--" if f == "SWOT truth"
                    else ("-." if f == "NorKyst"
                          else (":" if f == "DUACS block-repeat" else "-")))
            for f in FIELDS}


STYLE = _mkstyle()


def enable_norkyst():
    """Append NorKyst to the shared field list, in place, once."""
    if "NorKyst" not in FIELDS:
        FIELDS.append("NorKyst")
        STYLE.clear()
        STYLE.update(_mkstyle())
    return FIELDS
DATE_RE = re.compile(r"Expert_(\d+)_(\d+)_(\d{8})T")


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dp, dl = p2 - p1, np.deg2rad(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * RE * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def spacings(lat, lon):
    """Along-track (axis 0) and across-track (axis 1) pixel spacing, in metres.
    The equivalent of their SWOT_dist_angle<N>.mat, computed from the coordinates."""
    dal = np.empty_like(lat, dtype=np.float64)
    dal[:-1] = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])
    dal[-1] = dal[-2]
    dac = np.empty_like(lat, dtype=np.float64)
    dac[:, :-1] = haversine(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:])
    dac[:, -1] = dac[:, -2]
    return dal, dac


def read_pass(path):
    with Dataset(path) as d:
        lat = np.ma.filled(d["latitude"][:].astype(np.float64), np.nan)
        lon = np.ma.filled(d["longitude"][:].astype(np.float64), np.nan)
        ssha = np.ma.filled(d["ssha_filtered"][:].astype(np.float64), np.nan)
    lon = np.where(lon > 180, lon - 360, lon)
    ssha[np.abs(ssha) > 1e3] = np.nan          # belt and braces on the fill
    return lat, lon, ssha


def wang_window(ny, n_width=5, nn=9):
    """Their composite of `nn` overlapping Hanning windows -- flat interior, tapered
    ends.  Reproduced rather than replaced now that segments are long enough for it
    to mean something."""
    wdy = int(np.floor(ny / n_width))
    if wdy < 4:
        return np.hanning(ny)
    starts = (np.round(np.linspace(wdy / 2.0, ny - wdy / 2.0 - 1, nn))
              - round(wdy / 2.0) + 1).astype(int)
    w = np.hanning(wdy)
    W = np.zeros(ny)
    for s in starts:
        s = max(s - 1, 0)
        e = min(s + wdy, ny)
        W[s:e] += w[:e - s]
    return W / max(W.max(), 1e-12)


def shells(lam_min, lam_max, sca=SCA):
    """Fixed ladder of cut-off wavenumbers.

    Defined from a fixed wavelength range rather than from whichever pass happens to
    come first: passes differ in how many lines cross the box, and a ladder tied to
    one pass would leave the largest-scale bands unresolved -- and therefore near
    zero -- for every shorter pass, biasing the mean toward zero at large scales.
    Passes too short to resolve lam_max are skipped instead.
    """
    m = int(np.log(lam_max / lam_min) / np.log(sca)) + 1
    return np.sort(2 * np.pi / (lam_min * sca ** np.arange(m)))


def psd_along(field, d_along, win):
    a = (field - np.nanmean(field, axis=0, keepdims=True)) * win[:, None]
    F = np.fft.rfft(np.nan_to_num(a), axis=0)
    return (np.abs(F) ** 2).mean(axis=1) * 2.0 * d_along / (win ** 2).sum()


def shell_flux(u, v, d_along, dac, kcut, win):
    """T(k1, k2) in m^2/s^3, filtering along axis 0 (along-track)."""
    ny, nx = u.shape
    uw = (u - np.nanmean(u, axis=0, keepdims=True)) * win[:, None]
    vw = (v - np.nanmean(v, axis=0, keepdims=True)) * win[:, None]
    uw, vw = np.nan_to_num(uw), np.nan_to_num(vw)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=d_along)
    fu, fv = np.fft.fft(uw, axis=0), np.fft.fft(vw, axis=0)

    ns = len(kcut)
    U = np.empty((ns, ny, nx))
    V = np.empty((ns, ny, nx))
    for s, k in enumerate(kcut):
        m = (np.abs(ky) < k)[:, None]
        U[s] = np.fft.ifft(fu * m, axis=0).real
        V[s] = np.fft.ifft(fv * m, axis=0).real
    dU, dV = np.diff(U, axis=0), np.diff(V, axis=0)
    nb = ns - 1

    T = np.empty((nb, nb))
    for b2 in range(nb):
        u0, v0 = dU[b2], dV[b2]
        ux = np.gradient(u0, axis=1) / dac
        uy = np.gradient(u0, axis=0) / d_along
        vx = np.gradient(v0, axis=1) / dac
        vy = np.gradient(v0, axis=0) / d_along
        au, av = uw * ux + vw * uy, uw * vx + vw * vy
        for b1 in range(nb):
            T[b1, b2] = -np.nanmean(dU[b1] * au + dV[b1] * av)
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swath", default="swath_geometry")
    ap.add_argument("--archive", default="archive_swath")
    ap.add_argument("--out", default="runs/swath")
    ap.add_argument("--min-lines", type=int, default=256)
    ap.add_argument("--min-cols", type=int, default=8)
    ap.add_argument("--max-passes", type=int, default=0, help="0 = all")
    ap.add_argument("--lam-min-km", type=float, default=4.0)
    ap.add_argument("--lam-max-km", type=float, default=512.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mean, std = load_stats()

    arch = {}
    for f in sorted(Path(args.archive).glob("pred_*.npz")):
        d = np.load(f, allow_pickle=True)
        arch[str(d["date"])] = f
    print(f"{len(arch)} archived days")

    lat_f = C.LAT0 + (np.arange(C.NLAT_F) + 0.5) * C.DLAT_C / C.REFINE_LAT
    lon_f = C.LON0 + (np.arange(C.NLON_F) + 0.5) * C.DLON_C / C.REFINE_LON
    lat_c = C.coarse_lat(np.arange(C.NLAT_C))
    lon_c = C.coarse_lon(np.arange(C.NLON_C))
    sla_cache = np.load(C.CACHE_SLA, mmap_mode="r")

    files = sorted(Path(args.swath).rglob("*.nc"))
    if args.max_passes:
        files = files[:args.max_passes]
    lat0, lat1 = C.LAT0, C.LAT0 + C.NLAT_C * C.DLAT_C
    lon0, lon1 = C.LON0, C.LON0 + C.NLON_C * C.DLON_C

    kfix = shells(args.lam_min_km * 1000.0, args.lam_max_km * 1000.0)
    nb = len(kfix) - 1
    print(f"shell ladder {args.lam_min_km:.0f}-{args.lam_max_km:.0f} km, "
          f"{nb} bands (ratio {SCA})")
    accT, accP, accE, perT = {}, {}, {}, {f: [] for f in FIELDS}
    npass = 0

    reject = {"date": 0, "short": 0, "cols": 0}
    L = int(np.ceil(args.lam_max_km * 1000.0 / 2000.0))   # lines at 2 km posting
    print(f"window {L} lines (~{args.lam_max_km:.0f} km along-track), "
          f"disjoint windows per pass")

    # Group passes by date so the collocation interpolators, which are the expensive
    # part, are built once per day instead of once per window.
    by_date = {}
    for path in files:
        m = DATE_RE.search(path.name)
        d = f"{m.group(3)[:4]}-{m.group(3)[4:6]}-{m.group(3)[6:]}"
        by_date.setdefault(d, []).append(path)

    accT, accP, accE = {}, {}, {}
    perPass = {f: [] for f in FIELDS}
    npass, nwin = 0, 0

    for date in sorted(by_date):
        if date not in arch:
            reject["date"] += len(by_date[date])
            continue
        z = np.load(arch[date], allow_pickle=True)
        t = int(z["t"])
        ens = z["ens"].astype(np.float32) * std          # metres anomaly
        mu = z["mu"].astype(np.float32) * std
        sla_m = np.asarray(sla_cache[t], np.float32) - mean
        blockrep = np.repeat(np.repeat(np.nan_to_num(sla_m), C.REFINE_LAT, 0),
                             C.REFINE_LON, 1)

        def interp(grid, la, lo):
            return RegularGridInterpolator(
                (la, lo), np.nan_to_num(grid).astype(np.float64), method="cubic",
                bounds_error=False, fill_value=np.nan)

        GI = {
            "DUACS": interp(sla_m, lat_c, lon_c),
            "DUACS block-repeat": interp(blockrep, lat_f, lon_f),
            "mu (deterministic)": interp(mu, lat_f, lon_f),
            "diffusion member": interp(ens[0], lat_f, lon_f),
            "diffusion mean of 2": interp(ens[:2].mean(0), lat_f, lon_f),
            "diffusion mean of 4": interp(ens[:4].mean(0), lat_f, lon_f),
            "diffusion mean of 8": interp(ens.mean(0), lat_f, lon_f),
        }

        for path in by_date[date]:
            lat_a, lon_a, ssha_a = read_pass(path)
            inbox = ((lat_a >= lat0) & (lat_a <= lat1)
                     & (lon_a >= lon0) & (lon_a <= lon1))
            ok = inbox & np.isfinite(ssha_a)
            if ok.shape[0] < L or not ok.any():
                reject["short"] += 1
                continue
            c = np.pad(np.cumsum(ok, axis=0, dtype=np.int32), ((1, 0), (0, 0)))
            full = (c[L:] - c[:-L]) == L

            # Every DISJOINT window along the pass, not just the single best one.
            # Windows are kept disjoint so they stay independent; the standard error
            # below is still taken over passes, since windows within one pass share
            # a day and a region and would understate it.
            got = []
            for i0 in range(0, full.shape[0], L):
                row = full[i0].astype(np.int8)
                d = np.diff(np.concatenate([[0], row, [0]]))
                st, en = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
                if st.size == 0:
                    continue
                bi = int(np.argmax(en - st))
                cols = np.arange(st[bi], en[bi])
                if cols.size >= args.min_cols:
                    got.append((i0, cols))
            if not got:
                reject["cols"] += 1
                continue

            win_T = {f: [] for f in FIELDS}
            for (i0, cols) in got:
                sl = slice(i0, i0 + L)
                la = lat_a[sl][:, cols]
                lo = lon_a[sl][:, cols]
                ss = ssha_a[sl][:, cols]
                ny, nx = ss.shape
                d_al, d_ac = spacings(la, lo)
                d_along = float(np.nanmedian(d_al))
                f_cor = 2 * OMEGA * np.sin(np.deg2rad(la))
                pts = np.stack([la.ravel(), lo.ravel()], axis=1)

                F = {k: GI[k](pts).reshape(ny, nx) for k in GI}
                F["SWOT truth"] = ss
                if not np.isfinite(F["mu (deterministic)"]).all():
                    keep = np.isfinite(F["mu (deterministic)"]).all(axis=0)
                    if keep.sum() < args.min_cols:
                        continue
                    F = {k: v[:, keep] for k, v in F.items()}
                    la, d_al, d_ac = la[:, keep], d_al[:, keep], d_ac[:, keep]
                    f_cor = f_cor[:, keep]
                    ny, nx = F["SWOT truth"].shape
                win = wang_window(ny)

                for name in FIELDS:
                    eta = F[name]
                    u = -(G / f_cor) * np.gradient(eta, axis=0) / d_al
                    v = (G / f_cor) * np.gradient(eta, axis=1) / d_ac
                    T = shell_flux(u, v, d_along, d_ac, kfix, win)
                    win_T[name].append(T)
                    p_ = psd_along(eta, d_along, win)
                    e_ = psd_along(u, d_along, win) + psd_along(v, d_along, win)
                    accP[name] = p_ if name not in accP else accP[name] + p_
                    accE[name] = e_ if name not in accE else accE[name] + e_
                    accT[name] = T if name not in accT else accT[name] + T
                nwin += 1

            if not win_T[FIELDS[0]]:
                reject["cols"] += 1
                continue
            for name in FIELDS:
                perPass[name].append(np.mean(win_T[name], axis=0))
            npass += 1
            print(f"  {path.name[:44]}  {date}  {len(got)} windows x "
                  f"{L} lines", flush=True)

    for f in FIELDS:
        accT[f] /= max(nwin, 1)
        accP[f] /= max(nwin, 1)
        accE[f] /= max(nwin, 1)

    print(f"\nused {npass} passes, {nwin} windows; rejected: {reject}")
    if npass == 0:
        raise SystemExit("no usable passes -- see the rejection counts above")
    kmid = np.sqrt(kfix[:-1] * kfix[1:]) / (2 * np.pi)
    K1, K2 = np.meshgrid(kmid, kmid, indexing="ij")
    loc = ((K1 / K2) <= RATIO) & ((K1 / K2) >= 1 / RATIO)

    def totals(T):
        Tr = T * RHO0
        return (sum(np.nansum(Tr[k:, :k] * loc[k:, :k]) for k in range(1, nb)),
                sum(np.nansum(Tr[k:, :k] * ~loc[k:, :k]) for k in range(1, nb)))

    print(f"\n{npass} passes, shells {2*np.pi/kfix[-1]/1000:.1f}"
          f"-{2*np.pi/kfix[0]/1000:.0f} km, {nb} bands")
    print("\n=== cross-scale EKE flux (10^-6 W/m^3), + = forward ===")
    print("    +- is the standard error over passes")
    print(f"  {'field':<22}{'local':>20}{'nonlocal':>20}{'total':>20}")
    curves = {}
    for f in FIELDS:
        a = 1e6 * np.array([totals(T) for T in perPass[f]])
        m_, se = a.mean(0), a.std(0, ddof=1) / np.sqrt(len(a))
        tot = a.sum(1)
        Tr = accT[f] * RHO0
        curves[f] = (np.array([np.nansum(Tr[k:, :k] * loc[k:, :k])
                               for k in range(1, nb)]),
                     np.array([np.nansum(Tr[k:, :k] * ~loc[k:, :k])
                               for k in range(1, nb)]))
        print(f"  {f:<22}{m_[0]:>12.3f} +-{se[0]:<6.3f}"
              f"{m_[1]:>12.3f} +-{se[1]:<6.3f}"
              f"{tot.mean():>12.3f} +-{tot.std(ddof=1)/np.sqrt(len(a)):<6.3f}")

    np.savez(out / "swath.npz", kmid=kmid, npass=npass,
             **{f"T_{f}": accT[f] for f in FIELDS},
             **{f"P_{f}": accP[f] for f in FIELDS},
             **{f"E_{f}": accE[f] for f in FIELDS})
    json.dump({f: {"local": curves[f][0].tolist(),
                   "nonlocal": curves[f][1].tolist()} for f in FIELDS},
              open(out / "transfer.json", "w"), indent=2)
    _plots(out, kfix, kmid, accP, accE, accT, curves, npass, nwin, nb,
           2000.0)


def _plots(out, kcut, kmid, P, E, T, curves, npass, nwin, nb, d_along):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = 2 * (len(P[FIELDS[0]]) - 1)
    freq = np.fft.rfftfreq(n, d=d_along)
    wl = np.where(freq > 0, 1 / np.maximum(freq, 1e-30), np.inf) / 1000.0

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    for f in FIELDS:
        ax[0].loglog(wl[1:], P[f][1:], label=f, **STYLE[f])
        ax[1].loglog(wl[1:], E[f][1:], label=f, **STYLE[f])
    for a_, t, u in ((ax[0], "SSH power spectral density", "m$^2$/(cycle/m)"),
                     (ax[1], "geostrophic EKE spectral density",
                      "(m/s)$^2$/(cycle/m)")):
        a_.set_xlabel("wavelength [km]"); a_.set_ylabel(u); a_.set_title(t)
        a_.invert_xaxis(); a_.grid(alpha=0.3, which="both"); a_.legend(fontsize=7.5)
    fig.suptitle(f"Along-track spectra in swath geometry, "
                 f"{nwin} windows from {npass} passes")
    fig.savefig(out / "psd.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    kk = kmid * 1000.0
    blim = max(abs(np.nanpercentile(T[f] * RHO0 * 1e6, [2, 98])).max()
               for f in FIELDS)
    ncol = 4
    nrow = int(np.ceil(len(FIELDS) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.4 * nrow),
                            constrained_layout=True)
    axs = np.atleast_1d(axs).ravel()
    for a_, f in zip(axs, FIELDS):
        c = a_.pcolormesh(kk, kk, T[f] * RHO0 * 1e6, cmap="RdBu_r",
                          vmin=-blim, vmax=blim, shading="auto")
        a_.plot(kk, kk, "k--", lw=1)
        a_.plot(kk, kk / RATIO, "g--", lw=1)
        a_.plot(kk, kk * RATIO, "g--", lw=1)
        a_.set_xscale("log"); a_.set_yscale("log")
        a_.set_xlim(kk[0], kk[-1]); a_.set_ylim(kk[0], kk[-1]); a_.set_aspect("equal")
        a_.set_xlabel("q (cpkm) donor"); a_.set_title(f, fontsize=10)
    for a_ in axs[len(FIELDS):]:
        a_.axis("off")
    axs[0].set_ylabel("k (cpkm) receiver")
    fig.colorbar(c, ax=axs.tolist(), fraction=0.02, label="10$^{-6}$ W/m$^3$")
    fig.suptitle("Shell-to-shell EKE transfer, swath geometry (time mean)")
    fig.savefig(out / "shell_to_shell.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Three panels: local, nonlocal, and their sum -- the net flux through each
    # wavenumber, which is the quantity a cascade argument actually rests on.
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)
    x = kmid[1:] * 1000.0
    for f in FIELDS:
        fl, fn = curves[f]
        ax[0].semilogx(x, 1e6 * fl, label=f, **STYLE[f])
        ax[1].semilogx(x, 1e6 * fn, label=f, **STYLE[f])
        ax[2].semilogx(x, 1e6 * (fl + fn), label=f, **STYLE[f])
    for a_, t in ((ax[0], "Local transfer (within a factor 2)"),
                  (ax[1], "Nonlocal transfer"),
                  (ax[2], "Total = local + nonlocal")):
        a_.axhline(0, color="k", lw=0.8, ls="--")
        a_.set_xlabel("wavenumber (cpkm)")
        a_.set_ylabel("transfer rate (10$^{-6}$ W/m$^3$)")
        a_.set_title(t); a_.grid(alpha=0.3, which="both"); a_.legend(fontsize=7.5)
        sec = a_.secondary_xaxis("top", functions=(lambda v: 1 / np.maximum(v, 1e-9),
                                                   lambda v: 1 / np.maximum(v, 1e-9)))
        sec.set_xlabel("wavelength [km]")
    fig.suptitle("Time-mean cross-scale EKE transfer, swath geometry "
                 "(positive = forward, toward small scales)")
    fig.savefig(out / "cross_scale_transfer.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}/psd.png, shell_to_shell.png, cross_scale_transfer.png")


if __name__ == "__main__":
    main()
