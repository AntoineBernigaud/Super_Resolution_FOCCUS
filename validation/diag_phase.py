"""Is the transfer excess an ENERGY problem or a PHASE problem?

`whitened + sigma_max 13` matches its target's 10-60 km power (MD/target ~ 1.0) and
still runs ~8x hot in local transfer (11.78 +-1.89 against the truth's 1.49 +-2.28).
Transfer is a triple product, so it is not constrained by marginal statistics: a field
can carry exactly the right power in every band and still flux energy between them at
the wrong rate, if the PHASE RELATIONS between bands are wrong.  This measures that
directly.

THE SURROGATE.  For every window and every field, the transfer is recomputed on a
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


def _plots(report, out):
    """The surrogate test: real transfer against the phase-randomised copy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, rep in report.items():
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
                    help="cost is per field: `--surrogates` phase-randomised copies "
                         "are drawn for each, so a 5-field run costs ~2.5x a 2-field "
                         "one.  DUACS and mu are the controls -- both are nearly "
                         "devoid of real signal below ~40 km.")
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


    _plots(report, out)

    (out / "phase.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nwrote {out}/phase.json")


if __name__ == "__main__":
    main()
