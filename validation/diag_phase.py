"""Is the transfer excess an ENERGY problem or a PHASE problem?

`whitened + sigma_max 13` matches its target's 10-60 km power (MD/target ~ 1.0) and
still runs ~8x hot in local transfer (11.78 +-1.89 against the truth's 1.49 +-2.28).
Transfer is a triple product, so it is not constrained by marginal statistics: a field
can carry exactly the right power in every band and still flux energy between them at
the wrong rate, if the PHASE RELATIONS between bands are wrong.  This measures that
directly, two ways.

1. THE SURROGATE.  For every window and every field, the transfer is recomputed on a
   phase-randomised copy: the 2-D FFT amplitude of eta is preserved EXACTLY and the
   phases are replaced by those of a Gaussian field.  The surrogate therefore has the
   identical power spectrum -- identical PSD, identical EKE, identical everything
   marginal -- and no phase organisation at all.  Reading:

     T_surrogate ~ 0 for every field   -> ALL transfer is phase organisation, and the
                                          model's excess is entirely a phase defect
     T_surrogate large                 -> the spectrum shape alone drives transfer and
                                          the excess is partly an energy defect

   Taking the phases from `fft2` of a real Gaussian field, rather than drawing angles
   directly, is what guarantees the Hermitian symmetry that keeps the surrogate real.
   Geostrophy is applied AFTER randomisation, so u, v stay consistent with eta and the
   only thing destroyed is phase.

2. BICOHERENCE.  The normalised bispectrum along-track,

       b^2(k1,k2) = |E[F(k1) F(k2) F*(k1+k2)]|^2
                    / ( E[|F(k1) F(k2)|^2] . E[|F(k1+k2)|^2] )

   is amplitude-independent by construction and lies in [0, 1]: 0 means the three
   modes have independent phases, 1 means they are locked.  It is the scale-by-scale
   statement of the same question, and unlike the transfer it cannot be moved by
   getting the energy wrong.  Reported over LOCAL triads (k1 and k2 within RATIO of
   each other), which is the band the local transfer sums over.

Windows, geometry, shells and the window function are taken from `swath_splits` /
`swath_transfer` unchanged, so the numbers sit on the same footing as the transfer
table they are explaining.
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
from swath_transfer import (FIELDS, RATIO, RHO0, G, OMEGA,
                            spacings, wang_window, shells, shell_flux)


def phase_randomise(eta, rng):
    """Same 2-D power spectrum, independent phases.  Real by construction."""
    F = np.fft.fft2(eta)
    Fr = np.fft.fft2(rng.standard_normal(eta.shape))
    return np.fft.ifft2(np.abs(F) * np.exp(1j * np.angle(Fr))).real


def bicoherence_acc(eta, win, acc, nb_keep):
    """Accumulate bispectrum numerator and the two denominators, per column."""
    ny = eta.shape[0]
    z = (eta - eta.mean(axis=0, keepdims=True)) * win[:, None]
    F = np.fft.rfft(z, axis=0)[:nb_keep]                       # (nb_keep, ncol)
    i = np.arange(nb_keep)
    s = i[:, None] + i[None, :]
    ok = s < nb_keep
    s_c = np.where(ok, s, 0)
    for c in range(F.shape[1]):
        f = F[:, c]
        M = f[:, None] * f[None, :]
        F3 = f[s_c]
        acc["num"] += np.where(ok, M * np.conj(F3), 0)
        acc["d12"] += np.where(ok, np.abs(M) ** 2, 0)
        acc["d3"] += np.where(ok, np.abs(F3) ** 2, 0)
        acc["n"] += 1



def _plots(report, out):
    """Bicoherence and the surrogate test.  Until 2026-09-16 this script wrote only
    JSON, so the one diagnostic CLAUDE.md calls "the instrument to use" was never
    actually looked at.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, rep in report.items():
        wl = np.asarray(rep["bicoherence_lambda_km"], float)
        bic, berr = rep["bicoherence"], rep["bicoherence_err"]
        if "bicoherence_dilution" in rep:
            dil = np.asarray(rep["bicoherence_dilution"], float)
        else:
            # Reconstructed EXACTLY for reports written before the dilution curve was
            # stored: nb_keep is the length of the wavelength axis and RATIO is a
            # module constant, so the local mask and the i+j >= nb_keep cut are both
            # recoverable.  Without this an old report would replot with no shading,
            # which is the misleading figure this shading exists to prevent.
            nbk = len(wl)
            ii = np.arange(nbk)
            R = np.where(ii[None, :] > 0, ii[:, None] / np.maximum(ii[None, :], 1), 0.0)
            loc = (R <= RATIO) & (R >= 1 / RATIO) & (ii[:, None] > 0)
            S = ii[:, None] + ii[None, :]
            dil = np.array([float((S[r][loc[r]] >= nbk).mean())
                            if loc[r].any() else np.nan for r in range(nbk)])
        fields = list(bic)

        fig, ax = plt.subplots(figsize=(9, 5.6))
        # Shade where the row average is diluted by structural zeros.  Two levels: a
        # light band where it has started and a hatched one where the row is mostly
        # zeros and carries no information at all.
        for thr, al, hatch, lab in ((0.05, 0.10, None, "diluted >5% by structural zeros"),
                                    (0.50, 0.18, "///", "row is mostly structural zeros")):
            bad = np.isfinite(dil) & (dil > thr) & (wl > 0)
            if bad.any():
                ax.axvspan(np.nanmin(wl[bad]), np.nanmax(wl[bad]), color="0.5",
                           alpha=al, hatch=hatch, lw=0, label=lab, zorder=0)
        for k in fields:
            c = np.asarray(bic[k], float)
            e = np.asarray(berr[k], float)
            g = np.isfinite(c) & (wl > 0) & (c > 0)
            col = "k" if "truth" in k.lower() else None
            ln, = ax.plot(wl[g], c[g], lw=1.8, color=col,
                          label=k + (" (reference)" if "truth" in k.lower() else ""))
            ax.fill_between(wl[g], np.maximum(c[g] - e[g], 1e-6), c[g] + e[g],
                            color=ln.get_color(), alpha=0.22, lw=0)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("wavelength [km]")
        ax.set_ylabel(r"mean bicoherence $b^2$ over local triads")
        ax.set_title(f"{name} -- phase coupling between scales\n"
                     "amplitude-independent by construction: it cannot be moved by "
                     "getting the energy wrong", fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
        fig.tight_layout()
        f = out / f"bicoherence_{name}.png"
        fig.savefig(f, dpi=130); plt.close(fig)
        print(f"wrote {f}")

        # the surrogate test: real transfer against the phase-randomised copy
        keys = [k for k in rep if isinstance(rep[k], dict) and "real" in rep[k]]
        if not keys:
            continue
        fig, ax = plt.subplots(figsize=(min(9, 2.6 + 1.9 * len(keys)), 4.8))
        xs = np.arange(len(keys))
        for off, which, col, lab in ((-0.18, "real", "tab:blue", "real field"),
                                     (0.18, "surrogate", "0.6",
                                      "phase-randomised (same PSD, same EKE)")):
            v = [float(rep[k][which][0]) for k in keys]
            e = [float(rep[k][which][1]) for k in keys]
            ax.bar(xs + off, v, 0.34, yerr=e, capsize=4, color=col, label=lab)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(xs); ax.set_xticklabels(keys, fontsize=9)
        ax.set_ylabel("local cross-scale transfer")
        ax.set_title(f"{name} -- surrogate test: every surrogate consistent with zero "
                     "means\nall of the transfer is phase organisation", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        f = out / f"transfer_surrogate_{name}.png"
        fig.savefig(f, dpi=130); plt.close(fig)
        print(f"wrote {f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archives", nargs="+", default=None,   # required unless --replot
                    help="name=dir, as swath_splits takes them.  The special value "
                         "name=RECORD scores the TRUTH ONLY over every day in the "
                         "record that has swath geometry -- no archive, no model -- "
                         "which is how the truth's own cascade gets enough windows "
                         "to be measurable at all.")
    ap.add_argument("--surrogates", type=int, default=3)
    ap.add_argument("--max-days", type=int, default=0)
    ap.add_argument("--min-cols", type=int, default=8)
    ap.add_argument("--lam-min-km", type=float, default=4.0)
    ap.add_argument("--lam-max-km", type=float, default=512.0)
    ap.add_argument("--fields", nargs="+",
                    default=["SWOT truth", "DUACS", "mu (deterministic)",
                             "diffusion member", "diffusion mean of 8"],
                    help="cost is per field -- surrogates are drawn for each -- so a "
                         "5-field run is ~2.5x a 2-field one.  DUACS and mu are the "
                         "CONTROLS and are the reason to pay it: both are nearly "
                         "devoid of real signal below ~40 km, so they show what this "
                         "estimator returns when the band is empty -- DUACS reads "
                         "0.0153 at 25 km, five times the member and 22x the truth, "
                         "with a 48% error bar.  Quote no bicoherence excess without "
                         "them.  `diffusion mean of 8` does NOT behave like the "
                         "cascade under averaging (transfer 18.6 -> 1.3): b^2 is "
                         "normalised, so averaging shrinks numerator and denominator "
                         "together and the ratio need not fall -- measured 0.0045 at "
                         "25 km, ABOVE the member's 0.0029.")
    ap.add_argument("--replot", action="store_true",
                    help="rebuild the figures from an existing phase.json in --out and "
                         "exit, without redoing the surrogates (the expensive part)")
    ap.add_argument("--out", default="validation/plots/lam1.0/phase")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.replot and not args.archives:
        raise SystemExit("--archives is required unless --replot is given")
    if args.replot:
        f = out / "phase.json"
        if not f.exists():
            raise SystemExit(f"no {f} to replot")
        _plots(json.loads(f.read_text()), out)
        return
    mean, std = load_stats()
    rng = np.random.default_rng(0)

    kfix = shells(args.lam_min_km * 1000.0, args.lam_max_km * 1000.0)
    nb = len(kfix) - 1
    kmid = np.sqrt(kfix[:-1] * kfix[1:])
    K1, K2 = np.meshgrid(kmid, kmid, indexing="ij")
    loc = ((K1 / K2) <= RATIO) & ((K1 / K2) >= 1 / RATIO)

    L = int(np.ceil(args.lam_max_km * 1000.0 / 2000.0))
    lat_f = C.LAT0 + (np.arange(C.NLAT_F) + 0.5) * C.DLAT_C / C.REFINE_LAT
    lon_f = C.LON0 + (np.arange(C.NLON_F) + 0.5) * C.DLON_C / C.REFINE_LON
    lat_c, lon_c = C.coarse_lat(np.arange(C.NLAT_C)), C.coarse_lon(np.arange(C.NLON_C))
    ssha_cache = np.load(C.CACHE_SSHA, mmap_mode="r")
    sla_cache = np.load(C.CACHE_SLA, mmap_mode="r")
    lat0, lat1 = C.LAT0, C.LAT0 + C.NLAT_C * C.DLAT_C
    lon0, lon1 = C.LON0, C.LON0 + C.NLON_C * C.DLON_C

    # A local triad needs i + j < nb_keep with j >= i/RATIO, so nb_keep caps the
    # SMALLEST resolvable wavelength at about nb_keep/(1+1/RATIO) bins -- at 48 that
    # was bin 32, lambda ~ 16 km, and every row below it came out as an exact 0.0000
    # that looked like a measurement of "no phase coupling" and was really an empty
    # average.  Keep all of them.
    nb_keep = L // 2 + 1
    report = {}

    for spec in args.archives:
        name, d = spec.split("=", 1)
        record = (d == "RECORD")
        if record:
            dates = np.load(C.PATCH_INDEX)["dates"].astype("datetime64[D]")
            files = list(range(len(dates)))
            fields = ["SWOT truth"]
        else:
            files = sorted(Path(d).glob("pred_*.npz"))
            fields = args.fields
        if args.max_days:
            files = files[:args.max_days]
        if not files:
            print(f"[{name}] no archive at {d}")
            continue

        realT = {f: [] for f in fields}
        surrT = {f: [] for f in fields}
        bic = {f: dict(num=np.zeros((nb_keep, nb_keep), complex),
                       d12=np.zeros((nb_keep, nb_keep)),
                       d3=np.zeros((nb_keep, nb_keep)), n=0) for f in fields}
        # per-window copies, for a jackknife: b^2 is a ratio of averages, so its
        # error cannot be taken from a per-window standard deviation
        bic_w = {f: [] for f in fields}
        nwin = 0

        for fp in files:
            if record:
                t = int(fp)
                date = str(dates[t])
                ens = mu = None
            else:
                z = np.load(fp, allow_pickle=True)
                date, t = str(z["date"]), int(z["t"])
                ens = z["ens"].astype(np.float32) * std
                mu = z["mu"].astype(np.float32) * std
            sla_m = np.asarray(sla_cache[t], np.float32) - mean
            y = np.asarray(ssha_cache[t], np.float32)
            yv = np.isfinite(y)
            yc = np.clip(np.nan_to_num(y, nan=0.0), -C.CLIP_M, C.CLIP_M) - mean

            def rgi(g, la, lo, method="cubic"):
                return RegularGridInterpolator(
                    (la, lo), np.nan_to_num(g).astype(np.float64), method=method,
                    bounds_error=False, fill_value=np.nan)

            GI = {"DUACS": rgi(sla_m, lat_c, lon_c)}
            if ens is not None:
                GI["mu (deterministic)"] = rgi(mu, lat_f, lon_f)
                GI["diffusion member"] = rgi(ens[0], lat_f, lon_f)
                GI["diffusion mean of 8"] = rgi(ens.mean(0), lat_f, lon_f)
            GT = rgi(yc, lat_f, lon_f)
            GM = rgi(yv.astype(np.float64), lat_f, lon_f, method="linear")

            for _pass, gfile in SG.passes_for(date):
                lat_a, lon_a = SG.geometry(gfile)
                inbox = ((lat_a >= lat0) & (lat_a <= lat1)
                         & (lon_a >= lon0) & (lon_a <= lon1))
                if not inbox.any():
                    continue
                rows = np.nonzero(inbox.any(axis=1))[0]
                la_p, lo_p = lat_a[rows[0]:rows[-1] + 1], lon_a[rows[0]:rows[-1] + 1]
                ib = inbox[rows[0]:rows[-1] + 1]
                pts = np.stack([la_p.ravel(), lo_p.ravel()], axis=1)
                truth = np.where((GM(pts).reshape(la_p.shape) > 0.999) & ib,
                                 GT(pts).reshape(la_p.shape), np.nan)
                for (i0, cols) in pack_windows(np.isfinite(truth), L, args.min_cols):
                    sl = slice(i0, i0 + L)
                    la, lo = la_p[sl][:, cols], lo_p[sl][:, cols]
                    ny, nx = la.shape
                    d_al, d_ac = spacings(la, lo)
                    d_along = float(np.nanmedian(d_al))
                    f_cor = 2 * OMEGA * np.sin(np.deg2rad(la))
                    q = np.stack([la.ravel(), lo.ravel()], axis=1)
                    F = {k: GI[k](q).reshape(ny, nx) for k in GI}
                    F["SWOT truth"] = truth[sl][:, cols]
                    if not all(np.isfinite(F[k]).all() for k in fields):
                        continue
                    win = wang_window(ny)

                    def flux(eta):
                        u = -(G / f_cor) * np.gradient(eta, axis=0) / d_al
                        v = (G / f_cor) * np.gradient(eta, axis=1) / d_ac
                        return shell_flux(u, v, d_along, d_ac, kfix, win)

                    for k in fields:
                        eta = F[k]
                        realT[k].append(flux(eta))
                        surrT[k].append(np.mean(
                            [flux(phase_randomise(eta, rng))
                             for _ in range(args.surrogates)], axis=0))
                        one = dict(num=np.zeros((nb_keep, nb_keep), complex),
                                   d12=np.zeros((nb_keep, nb_keep)),
                                   d3=np.zeros((nb_keep, nb_keep)), n=0)
                        bicoherence_acc(eta, win, one, nb_keep)
                        for key in ("num", "d12", "d3"):
                            bic[k][key] += one[key]
                        bic[k]["n"] += one["n"]
                        bic_w[k].append(one)
                    nwin += 1
            print(f"  [{name}] {date}  windows {nwin}", flush=True)

        if not nwin:
            print(f"[{name}] no usable windows")
            continue

        def loc_total(T):
            Tr = T * RHO0 * 1e6
            return sum(np.nansum(Tr[k:, :k] * loc[k:, :k]) for k in range(1, nb))

        print(f"\n=== {name}: {nwin} windows, {args.surrogates} surrogates each ===")
        print(f"  {'field':<24}{'local T':>12}{'phase-random':>15}"
              f"{'randomised/real':>18}")
        report[name] = {}
        for k in fields:
            a = np.array([loc_total(T) for T in realT[k]])
            b = np.array([loc_total(T) for T in surrT[k]])
            se_a = a.std(ddof=1) / np.sqrt(len(a))
            se_b = b.std(ddof=1) / np.sqrt(len(b))
            report[name][k] = dict(real=[a.mean(), se_a], surrogate=[b.mean(), se_b])
            print(f"  {k:<24}{a.mean():>7.2f} +-{se_a:<4.2f}"
                  f"{b.mean():>10.2f} +-{se_b:<4.2f}"
                  f"{b.mean()/a.mean() if a.mean() else np.nan:>18.3f}")

        freq = np.fft.rfftfreq(L, d=d_along)[:nb_keep]
        wl = np.where(freq > 0, 1 / np.maximum(freq, 1e-30), np.inf) / 1000.0
        print(f"\n  === mean bicoherence b^2 over LOCAL triads ===")
        print(f"  {'lambda km':>10}" + "".join(f"{k[:14]:>16}" for k in fields))
        i = np.arange(nb_keep)
        R = np.where(i[None, :] > 0, i[:, None] / np.maximum(i[None, :], 1), 0.0)
        local = (R <= RATIO) & (R >= 1 / RATIO) & (i[:, None] > 0)
        curves = {}
        err = {}
        for k in fields:
            B = bic[k]

            def rowmean(num, d12, d3):
                b2 = np.abs(num) ** 2 / np.maximum(d12 * d3, 1e-300)
                return np.array([np.nanmean(b2[r][local[r]]) if local[r].any()
                                 else np.nan for r in range(nb_keep)])

            curves[k] = rowmean(B["num"], B["d12"], B["d3"])
            # delete-one jackknife over windows
            J = np.array([rowmean(B["num"] - o["num"], B["d12"] - o["d12"],
                                  B["d3"] - o["d3"]) for o in bic_w[k]])
            nj = len(J)
            err[k] = np.sqrt((nj - 1) / nj * np.nansum((J - J.mean(0)) ** 2, axis=0))
        valid = [r for r in range(2, nb_keep) if local[r].any()]
        for r in valid[::max(1, len(valid) // 14)]:
            print(f"  {wl[r]:10.1f}" + "".join(
                f"{curves[k][r]:10.4f}+-{err[k][r]:<5.4f}" for k in fields))
        report[name]["bicoherence"] = {k: curves[k].tolist() for k in fields}
        report[name]["bicoherence_err"] = {k: err[k].tolist() for k in fields}
        report[name]["bicoherence_lambda_km"] = wl.tolist()
        # How much of each row average is STRUCTURAL ZERO.  bicoherence_acc only fills
        # triads with i + j < nb_keep (`ok = s < nb_keep`), but `local` does not
        # exclude the rest, so they enter the mean as exact zeros and drag it down.
        # Measured here rather than guessed, and plotted, because a row that is mostly
        # structural zeros is NOT a measurement of "no phase coupling" -- an earlier
        # run with nb_keep = 48 printed a column of convincing 0.0000s for that reason.
        S = i[:, None] + i[None, :]
        dil = np.array([float((S[r][local[r]] >= nb_keep).mean())
                        if local[r].any() else np.nan for r in range(nb_keep)])
        report[name]["bicoherence_dilution"] = dil.tolist()

    _plots(report, out)

    (out / "phase.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nwrote {out}/phase.json")


if __name__ == "__main__":
    main()
