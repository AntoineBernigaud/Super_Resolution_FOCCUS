"""PSD and cross-scale EKE transfer per split, on SWOT's swath geometry.

The index CSV gives the passes for every date in the record, and geometry repeats
across cycles, so the 143 copied files describe the sampling on all 828 days.  That is
what makes this possible on val and test, not just the 23 days whose SSH we hold.

The truth is therefore the GRIDDED ssha collocated back onto the swath points, for
every split including train -- consistency across splits matters more than one split
getting the native L3 field.  It is a second interpolation, but the first one was an
interpolant onto a grid FINER than the 2 km source (the contract's point that this is
an upsample with no aliasing), so evaluating it back at the original sample locations
is second-order.  Below ~10 km I would not trust it; --native-check quantifies the
cost on the 23 days where both are available.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import config as C
import swath_geom as SG
from data import load_stats
import swath_transfer as _st
from swath_transfer import (FIELDS, STYLE, RATIO, RHO0, G, OMEGA,
                            spacings, wang_window, shells, psd_along, shell_flux)


def pack_windows(ok, L, min_cols):
    """All disjoint windows along a pass, scanning EVERY start offset.

    An earlier version only tested starts at multiples of L, which discarded passes
    whose one valid stretch happened to sit at an awkward offset -- 98 usable passes
    fell to 81.  Here every offset is a candidate and accepted windows are packed
    greedily so they stay disjoint and therefore independent.
    """
    if ok.shape[0] < L:
        return []
    c = np.pad(np.cumsum(ok, axis=0, dtype=np.int32), ((1, 0), (0, 0)))
    full = (c[L:] - c[:-L]) == L
    out, taken = [], np.zeros(full.shape[0], bool)
    order = np.argsort(-full.sum(axis=1))            # widest first
    for i0 in order:
        if full[i0].sum() < min_cols or taken[max(i0 - L + 1, 0):i0 + L].any():
            continue
        row = full[i0].astype(np.int8)
        d = np.diff(np.concatenate([[0], row, [0]]))
        st, en = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
        b = int(np.argmax(en - st))
        cols = np.arange(st[b], en[b])
        if cols.size < min_cols:
            continue
        taken[i0] = True
        out.append((int(i0), cols))
    return out


_BEST_PASS = None


def _best_pass(lat0, lat1, lon0, lon1, nk, lat_f, lon_f):
    """The geometry whose usable footprint holds the most points.

    With NorKyst on, "usable" also means inside its footprint: the best-in-box pass
    otherwise runs west of lon -1.85 where NorKyst has no data, and every window out
    there would be dropped -- losing the coverage this mode exists to buy.
    """
    global _BEST_PASS
    key = (lat0, lat1, lon0, lon1, nk is not None)
    if _BEST_PASS is not None and _BEST_PASS[0] == key:
        return _BEST_PASS[1]
    seen, best = set(), None
    for rows in SG.index().values():
        for _p, gfile, _s in rows:
            if gfile in seen:
                continue
            seen.add(gfile)
            try:
                la, lo = SG.geometry(gfile)
            except KeyError:
                continue
            okm = (la >= lat0) & (la <= lat1) & (lo >= lon0) & (lo <= lon1)
            if nk is not None:
                i = np.clip(np.searchsorted(lat_f, la) - 1, 0, len(lat_f) - 1)
                j = np.clip(np.searchsorted(lon_f, lo) - 1, 0, len(lon_f) - 1)
                okm &= nk.mask[i, j]
            n = int(okm.sum())
            if best is None or n > best[0]:
                best = (n, gfile)
    where = "box + NorKyst footprint" if nk is not None else "box"
    print(f"max-pass: scanned {len(seen)} geometries; using {best[1]} "
          f"with {best[0]:,} points inside the {where}")
    _BEST_PASS = (key, best[1])
    return best[1]


def analyse(archive, split, args, kfix, mean, std, grids):
    lat_f, lon_f, lat_c, lon_c, ssha_cache, sla_cache = grids
    L = int(np.ceil(args.lam_max_km * 1000.0 / 2000.0))
    lat0, lat1 = C.LAT0, C.LAT0 + C.NLAT_C * C.DLAT_C
    lon0, lon1 = C.LON0, C.LON0 + C.NLON_C * C.DLON_C

    accT, accP, accE = {}, {}, {}
    perPass = {f: [] for f in FIELDS}
    npass = nwin = 0

    files = sorted(Path(archive).glob("pred_*.npz"))
    if args.max_days:
        files = files[:args.max_days]
    for fp in files:
        z = np.load(fp, allow_pickle=True)
        date, t = str(z["date"]), int(z["t"])
        ens = z["ens"].astype(np.float32) * std
        mu = z["mu"].astype(np.float32) * std
        sla_m = np.asarray(sla_cache[t], np.float32) - mean
        blockrep = np.repeat(np.repeat(np.nan_to_num(sla_m), C.REFINE_LAT, 0),
                             C.REFINE_LON, 1)
        y = np.asarray(ssha_cache[t], np.float32)
        yv = np.isfinite(y)
        yc = (np.clip(np.nan_to_num(y, nan=0.0), -C.CLIP_M, C.CLIP_M) - mean)

        def rgi(g, la, lo, method="cubic"):
            return RegularGridInterpolator(
                (la, lo), np.nan_to_num(g).astype(np.float64), method=method,
                bounds_error=False, fill_value=np.nan)

        GI = {"DUACS": rgi(sla_m, lat_c, lon_c),
              "DUACS block-repeat": rgi(blockrep, lat_f, lon_f),
              "mu (deterministic)": rgi(mu, lat_f, lon_f),
              "diffusion member": rgi(ens[0], lat_f, lon_f),
              "diffusion mean of 2": rgi(ens[:2].mean(0), lat_f, lon_f),
              "diffusion mean of 4": rgi(ens[:4].mean(0), lat_f, lon_f),
              "diffusion mean of 8": rgi(ens.mean(0), lat_f, lon_f)}
        GNK = GNKM = None
        nk = getattr(args, "_nk", None)     # attached by main(); analyse() gets args
        if nk is not None:
            nkf = nk.field(t)
            if nkf is None:
                continue
            nkv = np.isfinite(nkf) & nk.mask
            GI["NorKyst"] = rgi(np.where(nkv, nkf, 0.0), lat_f, lon_f)
            # separate validity interpolator, for the same reason as GM below: rgi
            # nan_to_num's its input, so NorKyst outside its footprint would arrive
            # as a hard 0 rather than as missing, and the window would be accepted.
            GNKM = rgi(nkv.astype(np.float64), lat_f, lon_f, method="linear")
        GT = rgi(yc, lat_f, lon_f)
        # A separate mask interpolator: cubic-interpolating a NaN-filled truth would
        # silently invent zeros across the swath gaps.
        GM = rgi(yv.astype(np.float64), lat_f, lon_f, method="linear")

        passes = (SG.passes_for(date) if not getattr(args, "max_pass", False)
                  else [(0, _best_pass(lat0, lat1, lon0, lon1, nk, lat_f, lon_f))])
        for _pass, gfile in passes:
            lat_a, lon_a = SG.geometry(gfile)
            inbox = ((lat_a >= lat0) & (lat_a <= lat1)
                     & (lon_a >= lon0) & (lon_a <= lon1))
            if not inbox.any():
                continue
            rows = np.nonzero(inbox.any(axis=1))[0]
            r0, r1 = rows[0], rows[-1] + 1
            la_p, lo_p, ib = lat_a[r0:r1], lon_a[r0:r1], inbox[r0:r1]
            pts = np.stack([la_p.ravel(), lo_p.ravel()], axis=1)
            tv = GT(pts).reshape(la_p.shape)
            tm = GM(pts).reshape(la_p.shape)
            truth = np.where((tm > 0.999) & ib, tv, np.nan)
            # in max-pass mode the model fields exist everywhere, so a window only
            # has to lie in the box; requiring observed truth is what limits the
            # normal mode to the passes that actually flew
            ok = ib if getattr(args, "max_pass", False) else np.isfinite(truth)
            wins = pack_windows(ok, L, args.min_cols)
            if not wins:
                continue

            win_T = {f: [] for f in FIELDS}
            for (i0, cols) in wins:
                sl = slice(i0, i0 + L)
                la = la_p[sl][:, cols]
                lo = lo_p[sl][:, cols]
                ny, nx = la.shape
                d_al, d_ac = spacings(la, lo)
                d_along = float(np.nanmedian(d_al))
                f_cor = 2 * OMEGA * np.sin(np.deg2rad(la))
                q = np.stack([la.ravel(), lo.ravel()], axis=1)
                F = {k: GI[k](q).reshape(ny, nx) for k in GI}
                # In max-pass mode the truth is absent almost everywhere by
                # construction, and the finiteness check below tests every entry of
                # F -- so including it here rejected every window and the run
                # reported "no usable windows".
                if not getattr(args, "max_pass", False):
                    F["SWOT truth"] = truth[sl][:, cols]
                if GNKM is not None:
                    if not (GNKM(q).reshape(ny, nx) > 0.999).all():
                        continue
                if not all(np.isfinite(v).all() for v in F.values()):
                    continue
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
            if win_T[FIELDS[0]]:
                for name in FIELDS:
                    perPass[name].append(np.mean(win_T[name], axis=0))
                npass += 1
        print(f"  [{split}] {date}  passes so far {npass}, windows {nwin}",
              flush=True)

    if nwin == 0:
        return None
    for f in FIELDS:
        accT[f] /= nwin
        accP[f] /= nwin
        accE[f] /= nwin
    return dict(T=accT, P=accP, E=accE, perPass=perPass, npass=npass, nwin=nwin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--whole-plot", action="store_true",
                    help="also write psd_total_dataset.png, eke_psd_total_dataset.png "
                         "and cross_scale_transfer_total_dataset.png: TOTAL "
                         "transfer only (local + nonlocal), pooled over every archive "
                         "given.  There is a regime shift between train+val and test, "
                         "and the model is not trained against the transfer at all, "
                         "so the pooled curve is worth having on its own terms.")
    ap.add_argument("--archives", nargs="+",
                    default=["train=archive_train", "val=archive_val",
                             "test=archive"],
                    help="name=dir")
    ap.add_argument("--max-pass", action="store_true",
                    help="use ONE pass geometry -- the one covering the most points "
                         "in the box -- on EVERY day, instead of the passes that "
                         "actually crossed.  The model fields have no gaps, so each "
                         "day then contributes a full pass of windows rather than "
                         "whatever coverage allowed (318 windows over 40 days "
                         "normally, 25 once the NorKyst footprint is imposed, which "
                         "left the transfer error bars larger than the values).  "
                         "SWOT truth is necessarily dropped: it exists only where it "
                         "really flew, which is the point of doing this.")
    ap.add_argument("--out", default="runs/swath_splits")
    ap.add_argument("--lam-min-km", type=float, default=4.0)
    ap.add_argument("--lam-max-km", type=float, default=512.0)
    ap.add_argument("--min-cols", type=int, default=8)
    ap.add_argument("--max-days", type=int, default=0)
    ap.add_argument("--no-block-repeat", action="store_true",
                    help="drop the 'DUACS block-repeat' control curve from every "
                         "table and figure")
    args = ap.parse_args()

    if args.no_block_repeat and "DUACS block-repeat" in FIELDS:
        FIELDS.remove("DUACS block-repeat")
        STYLE.pop("DUACS block-repeat", None)

    if getattr(args, "max_pass", False):
        for drop in ("SWOT truth", "DUACS block-repeat"):
            if drop in FIELDS:
                FIELDS.remove(drop)
                STYLE.pop(drop, None)
        print("max-pass mode: SWOT truth and the block-repeat control are dropped")

    nk = None

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mean, std = load_stats()
    kfix = shells(args.lam_min_km * 1000.0, args.lam_max_km * 1000.0)
    nb = len(kfix) - 1
    kmid = np.sqrt(kfix[:-1] * kfix[1:]) / (2 * np.pi)
    K1, K2 = np.meshgrid(kmid, kmid, indexing="ij")
    loc = ((K1 / K2) <= RATIO) & ((K1 / K2) >= 1 / RATIO)
    print(f"shells {args.lam_min_km:.0f}-{args.lam_max_km:.0f} km, {nb} bands")

    grids = (C.LAT0 + (np.arange(C.NLAT_F) + 0.5) * C.DLAT_C / C.REFINE_LAT,
             C.LON0 + (np.arange(C.NLON_F) + 0.5) * C.DLON_C / C.REFINE_LON,
             C.coarse_lat(np.arange(C.NLAT_C)), C.coarse_lon(np.arange(C.NLON_C)),
             np.load(C.CACHE_SSHA, mmap_mode="r"),
             np.load(C.CACHE_SLA, mmap_mode="r"))

    res, summary = {}, {}
    for spec in args.archives:
        name, d = spec.split("=", 1)
        if not Path(d).exists() or not list(Path(d).glob("pred_*.npz")):
            print(f"[{name}] no archive at {d}, skipping")
            continue
        r = analyse(d, name, args, kfix, mean, std, grids)
        if r is None:
            print(f"[{name}] no usable windows")
            continue
        res[name] = r

        def totals(T):
            Tr = T * RHO0
            return (sum(np.nansum(Tr[k:, :k] * loc[k:, :k]) for k in range(1, nb)),
                    sum(np.nansum(Tr[k:, :k] * ~loc[k:, :k]) for k in range(1, nb)))

        print(f"\n=== {name}: {r['npass']} passes, {r['nwin']} windows ===")
        print(f"  {'field':<24}{'local':>18}{'nonlocal':>18}{'total':>18}")
        summary[name] = {}
        for f in FIELDS:
            a = 1e6 * np.array([totals(T) for T in r["perPass"][f]])
            m_, se = a.mean(0), a.std(0, ddof=1) / np.sqrt(len(a))
            tot = a.sum(1)
            summary[name][f] = dict(local=[m_[0], se[0]], nonlocal_=[m_[1], se[1]],
                                    total=[tot.mean(),
                                           tot.std(ddof=1) / np.sqrt(len(a))])
            print(f"  {f:<24}{m_[0]:>11.2f} +-{se[0]:<5.2f}"
                  f"{m_[1]:>11.2f} +-{se[1]:<5.2f}"
                  f"{tot.mean():>11.2f} +-{tot.std(ddof=1)/np.sqrt(len(a)):<5.2f}")

        Tr = {f: r["T"][f] * RHO0 for f in FIELDS}
        r["curves"] = {f: (np.array([np.nansum(Tr[f][k:, :k] * loc[k:, :k])
                                     for k in range(1, nb)]),
                           np.array([np.nansum(Tr[f][k:, :k] * ~loc[k:, :k])
                                     for k in range(1, nb)])) for f in FIELDS}

    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    np.savez(out / "swath_splits.npz", kmid=kmid,
             **{f"{s}_P_{f}": res[s]["P"][f] for s in res for f in FIELDS},
             **{f"{s}_E_{f}": res[s]["E"][f] for s in res for f in FIELDS},
             **{f"{s}_T_{f}": res[s]["T"][f] for s in res for f in FIELDS})
    _plots(out, res, kmid, args)
    if args.whole_plot:
        _plot_whole(out, res, kmid, nb)


def _plot_whole(out, res, kmid, nb):
    """PSD and TOTAL cross-scale transfer pooled over every archive given.

    Run with --archives train=... val=... test=... this is the whole dataset, which is
    the only sample large enough for the transfer to be better than sample-limited:
    321 test windows put the truth's local transfer at 1.88 +-2.18, while the full
    record gives 8.97 +-1.92.

    POOLING.  `analyse` divides its accumulators by that split's window count before
    returning, so res[s]["T"] and res[s]["P"] are MEANS per window, not sums.  Pooling
    them therefore needs the window-weighted mean, sum_s mean_s * nwin_s / sum_s
    nwin_s.  An earlier version summed the means and divided by the TOTAL window
    count, which is the same expression with the weights left out -- it under-scaled
    the pooled curve by roughly (total windows / number of splits), a factor of a few
    hundred here, and the resulting figure showed a cascade far weaker than any split.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(res)
    if not names:
        return
    nwin = sum(res[s]["nwin"] for s in names)
    if nwin == 0:
        return
    tag = " + ".join(f"{s} ({res[s]['nwin']})" for s in names) + f" = {nwin} windows"
    # kmid is CYCLES per metre -- the shell wavenumbers are divided by 2*pi -- so the
    # wavelength in km is 1/(k*1000), not 2*pi/(k*1000).
    lam = 1.0 / (kmid[1:] * 1000.0)
    pooled = lambda key, f_: (sum(res[s][key][f_] * res[s]["nwin"] for s in names)
                              / nwin)

    # --- 1. PSD over the whole dataset ---
    # P and E are along-track FFT spectra, NOT on the transfer's shell wavenumbers
    # (kmid): same axis construction as the per-split psd_by_split.png
    n_fft = 2 * (len(res[names[0]]["P"][FIELDS[0]]) - 1)
    freq = np.fft.rfftfreq(n_fft, d=2000.0)
    wl = np.where(freq > 0, 1 / np.maximum(freq, 1e-30), np.inf) / 1000.0
    for key, ylab, fname in (("P", "SSH PSD [m$^2$/cpm]", "psd_total_dataset"),
                             ("E", "geostrophic EKE PSD [m$^3$/s$^2$]",
                              "eke_psd_total_dataset")):
        fig, a = plt.subplots(figsize=(8.6, 6.4), constrained_layout=True)
        for f_ in FIELDS:
            a.loglog(wl[1:], pooled(key, f_)[1:], label=f_, **STYLE[f_])
        a.set_xlabel("wavelength [km]")
        a.set_ylabel(ylab)
        a.invert_xaxis()
        a.grid(alpha=0.3, which="both")
        a.legend(frameon=False, fontsize=8)
        a.set_title(f"{'SSH' if key == 'P' else 'EKE'} spectrum, TOTAL dataset\n"
                    + tag, fontsize=11, fontweight="bold")
        fig.savefig(out / f"{fname}.png", dpi=135, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}/{fname}.png  ({nwin} windows)")

    # --- 2. total cross-scale transfer over the whole dataset ---
    fig, a = plt.subplots(figsize=(8.6, 6.4), constrained_layout=True)
    hi = 0.0
    for f_ in FIELDS:
        T = pooled("T", f_) * RHO0
        tot = np.array([np.nansum(T[k:, :k]) for k in range(1, nb)]) * 1e6
        a.plot(lam, tot, label=f_, **STYLE[f_])
        if f_ in ("SWOT truth", "DUACS", "mu (deterministic)"):
            hi = max(hi, float(np.nanmax(np.abs(tot))))
    a.set_xscale("log")
    # symlog on y for the same reason the by-split panels use it: one run whose member
    # blows up would otherwise flatten every physical curve onto zero
    a.set_yscale("symlog", linthresh=max(hi, 1e-3))
    a.axhline(0.0, color="k", lw=0.8)
    a.invert_xaxis()
    a.set_xlabel("wavelength [km]")
    a.set_ylabel("total transfer (10$^{-6}$ W/m$^3$)")
    a.grid(alpha=0.3, which="both")
    a.legend(frameon=False, fontsize=8)
    a.set_title("Total cross-scale EKE transfer, TOTAL dataset\n" + tag,
                fontsize=11, fontweight="bold")
    fig.savefig(out / "cross_scale_transfer_total_dataset.png", dpi=135,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}/cross_scale_transfer_total_dataset.png  ({nwin} windows)")


def _plots(out, res, kmid, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(res)
    if not names:
        return
    n = 2 * (len(res[names[0]]["P"][FIELDS[0]]) - 1)
    freq = np.fft.rfftfreq(n, d=2000.0)
    wl = np.where(freq > 0, 1 / np.maximum(freq, 1e-30), np.inf) / 1000.0

    fig, axs = plt.subplots(2, len(names), figsize=(6.2 * len(names), 9.5),
                            constrained_layout=True, squeeze=False)
    for c, s in enumerate(names):
        for row, (key, t, u) in enumerate(
                ((("P"), "SSH PSD", "m$^2$/(cycle/m)"),
                 (("E"), "geostrophic EKE PSD", "(m/s)$^2$/(cycle/m)"))):
            a = axs[row][c]
            for f in FIELDS:
                a.loglog(wl[1:], res[s][key][f][1:], label=f, **STYLE[f])
            a.set_xlabel("wavelength [km]")
            a.set_ylabel(u)
            a.invert_xaxis()
            a.grid(alpha=0.3, which="both")
            a.set_title(f"{s} -- {t}  ({res[s]['nwin']} windows)", fontsize=10)
            if row == 0 and c == len(names) - 1:
                a.legend(fontsize=7)
    fig.suptitle("Along-track spectra in swath geometry, by split",
                 fontsize=13, fontweight="bold")
    fig.savefig(out / "psd_by_split.png", dpi=135, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(3, len(names), figsize=(6.2 * len(names), 13.5),
                            constrained_layout=True, squeeze=False)
    x = kmid[1:] * 1000.0
    for c, s in enumerate(names):
        for row, (idx, t) in enumerate(((0, "Local (within a factor 2)"),
                                        (1, "Nonlocal"),
                                        (2, "Total = local + nonlocal"))):
            a = axs[row][c]
            ref = []
            for f in FIELDS:
                fl, fn = res[s]["curves"][f]
                yv = fl if idx == 0 else (fn if idx == 1 else fl + fn)
                a.semilogx(x, 1e6 * yv, label=f, **STYLE[f])
                if f in ("SWOT truth", "DUACS", "mu (deterministic)"):
                    ref.append(np.abs(1e6 * yv).max())
            a.axhline(0, color="k", lw=0.8, ls="--")
            # A run whose member has blown up (the first loss_on_l3 run reached
            # -3221 against a truth of 1.5) would otherwise autoscale every other
            # curve, SWOT truth included, flat onto zero.  symlog keeps the
            # physical curves readable and still shows how far the outlier goes.
            lin = max(ref) if ref else 1.0
            a.set_yscale("symlog", linthresh=max(lin, 1e-9), linscale=1.4)
            a.axhspan(-lin, lin, color="0.5", alpha=0.07, zorder=0)
            a.set_xlabel("wavenumber (cpkm)")
            a.set_ylabel("transfer (10$^{-6}$ W/m$^3$)")
            a.set_title(f"{s} -- {t}", fontsize=10)
            a.grid(alpha=0.3, which="both")
            if row == 0 and c == len(names) - 1:
                a.legend(fontsize=7)
    fig.suptitle("Time-mean cross-scale EKE transfer by split "
                 "(positive = forward)", fontsize=13, fontweight="bold")
    fig.savefig(out / "cross_scale_transfer_by_split.png", dpi=135,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}/psd_by_split.png, cross_scale_transfer_by_split.png")


if __name__ == "__main__":
    main()
