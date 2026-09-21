"""Whiten the gridded target against native L3 instead of low-passing it.

`build_lowpass_target.py` removes the CloughTocher artifact by deleting everything
below ~15 km.  The measurement it cites says that is far too blunt.  From
`runs/native_vs_collocated/native_vs_collocated.json`, gridded/native SSH PSD is

    200 km 1.00 | 40 km 0.99 | 20 km 1.07 | 15 km 1.11 | 10 km 3.13 | 7 km 10.5 | 5 km 18.4

so the contamination is confined below ~12 km and is only 11% at 15 km, while the
15 km Gaussian keeps just 28% of the power at 15 km, 49% at 20 km and 82% at 40 km.
It throws away three to four times more real signal than artifact.  That, and not a
failure of the model, is why the low-pass run "declines to generate" below 20 km: it
is faithfully reproducing a target that stops there (target transfer^2 0.25 at 15 km,
model 0.23).

This builds the target that a matched filter would: multiply the Fourier amplitudes by

    H(k) = 1 / sqrt( P_gridded(k) / P_native(k) )

so the result has, by construction, the native L3 spectrum at EVERY scale -- nothing
is cut, the contaminated band is merely restored to its correct amplitude.  The model
can then be asked to reproduce the target's statistics, which is the point.

APPLYING IT TO A GAPPY FIELD.  A sharp spectral filter needs an FFT over a complete
rectangle, which this product does not have (see eke_transfer.py).  So H is fitted as a
sum of Gaussians, Sum_i a_i exp(-2 pi^2 sigma_i^2 / lambda^2) with Sum_i a_i = 1, and
each term is applied by normalised convolution, exactly as the low-pass build does.
Negative a_i are allowed and expected -- a difference of Gaussians is what makes the
1 -> 0.23 edge between 15 and 7 km sharp enough.  Sum a_i = 1 pins H(0) = 1 so the
mean and the mesoscale are untouched to machine precision.

BELOW 5 km THERE IS NO MEASUREMENT.  The probe ladder stops at 5 km.  Extrapolating
the ratio's log-log slope (-1.67 between 7 and 5 km) gives H ~ 0.19 at 4 km, i.e. a
plateau, not a roll-off, so `--floor-km` tapers H to zero below a stated wavelength
rather than inventing a correction there.  Default 4 km, just above the 3.48 km grid
Nyquist, so the taper is doing almost nothing; set it higher to be conservative.

CAVEAT worth stating before this is trained on: native L3 below ~10 km is itself
partly the KaRIn noise floor, so matching it means the model is allowed to generate
that too.  That is still 18x less energy at 5 km than the current target demands, and
unlike the low-pass it is an honest statement of what the instrument sees.
"""
import argparse
import json

import numpy as np
from scipy.ndimage import gaussian_filter

import config as C

DY_KM = C.DLAT_C / C.REFINE_LAT * 111.320          # 1.7394 km
NVC = C.ROOT / "runs/native_vs_collocated/native_vs_collocated.json"


def target_transfer(lam_km, ratio_lam, ratio, floor_km, taper_km):
    """Amplitude transfer H(lambda) wanted, from the measured PSD ratio."""
    # interpolate the ratio in log-log; clamp to >=1 so a scatter point below 1
    # (40 km reads 0.99) cannot ask for amplification
    # ratio_lam runs LARGE -> small, so the reversed arrays are ascending in
    # wavelength and `left` is the small-wavelength end (ratio[-1], the 5 km probe)
    # while `right` is the large one.  Getting these the wrong way round mirrored the
    # whole curve and only showed at the two endpoints, where argmin lands just
    # outside the probe range.
    r = np.exp(np.interp(np.log(lam_km), np.log(ratio_lam[::-1]),
                         np.log(ratio[::-1]), left=np.log(ratio[-1]),
                         right=np.log(ratio[0])))
    H = 1.0 / np.sqrt(np.maximum(r, 1.0))
    # no measurement below the last probe: taper to zero rather than extrapolate
    t = np.clip((lam_km - floor_km) / max(taper_km, 1e-6), 0.0, 1.0)
    return H * t * t * (3 - 2 * t)                  # smoothstep


def fit_gaussian_sum(lam, H, sig_km, ridge=1e-3):
    """Least squares a_i for Sum a_i exp(-2 pi^2 sig_i^2 / lam^2) = H, with Sum a_i = 1.

    The constraint is eliminated rather than penalised: a_0 = 1 - sum(a_1..) is
    substituted, so Sum a_i = 1 holds exactly whatever the fit does.

    The ridge term is not cosmetic.  An unpenalised fit drives the weights to
    +3.2/-3.1/+2.7 alternating, which is a difference of Gaussians with a long tail,
    and that tail overshoots to H = 1.017 at 40 km -- a 3.4% power ERROR in the band
    the whole exercise is trying to protect.  Penalising the coefficient norm buys a
    slightly softer edge at 7 km in exchange for a flat passband, which is the right
    trade here.  `lam` is log-spaced, so uniform weights already weight per octave.
    """
    B = np.exp(-2 * np.pi ** 2 * sig_km[None, :] ** 2 / lam[:, None] ** 2)
    A = B[:, 1:] - B[:, :1]
    rhs = H - B[:, 0]
    n = A.shape[1]
    G = A.T @ A + ridge * A.shape[0] * np.eye(n)
    rest = np.linalg.solve(G, A.T @ rhs)
    return np.concatenate([[1.0 - rest.sum()], rest])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(C.ROOT / "cache_ssha_wh.npy"))
    ap.add_argument("--floor-km", type=float, default=4.0)
    ap.add_argument("--taper-km", type=float, default=2.0)
    ap.add_argument("--sigmas-km", type=float, nargs="+",
                    default=[0.0, 0.6, 0.9, 1.2, 1.6, 2.2, 3.0, 4.2, 6.0, 9.0, 14.0])
    ap.add_argument("--ridge", type=float, default=3e-4)
    ap.add_argument("--fit-only", action="store_true",
                    help="print the transfer table and stop")
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--days", type=int, default=0, help="0 = whole record")
    args = ap.parse_args()

    d = json.loads(NVC.read_text())
    ratio_lam = np.array(d["probes_km"], float)
    ratio = np.array(d["ratio"]["SSH PSD"], float)

    lam = np.exp(np.linspace(np.log(3.5), np.log(400.0), 400))
    H = target_transfer(lam, ratio_lam, ratio, args.floor_km, args.taper_km)
    sig = np.array(args.sigmas_km, float)
    a = fit_gaussian_sum(lam, H, sig, args.ridge)
    fit = (np.exp(-2 * np.pi ** 2 * sig[None, :] ** 2 / lam[:, None] ** 2) * a).sum(1)

    print("Gaussian-sum fit to the artifact-free transfer")
    print(f"  {'sigma km':>10}{'weight':>10}")
    for s, w in zip(sig, a):
        print(f"  {s:10.2f}{w:+10.4f}")
    print(f"  sum of weights {a.sum():.6f}  (must be 1)")
    print(f"\n  {'lambda km':>10}{'wanted H':>10}{'fitted H':>10}{'lp15 H':>9}")
    s15 = float(np.sqrt(np.log(2.0) * 15.0 ** 2 / (2 * np.pi ** 2)))
    for L in (200, 100, 60, 40, 30, 20, 15, 12, 10, 7, 5, 4):
        i = int(np.argmin(np.abs(lam - L)))
        print(f"  {L:10.0f}{H[i]:10.3f}{fit[i]:10.3f}"
              f"{np.exp(-2*np.pi**2*s15**2/L**2):9.3f}")
    pb = lam >= 20.0
    print(f"\n  max |fitted - wanted|   passband (>=20 km) {np.abs(fit-H)[pb].max():.4f}"
          f"   overall {np.abs(fit-H).max():.3f}")
    bad = float(np.abs(fit - H)[pb].max())
    if bad > 0.08:
        print("  WARNING: the Gaussian ladder cannot represent this edge; add sigmas")

    if args.fit_only:
        return

    src = np.load(C.CACHE_SSHA, mmap_mode="r")
    nt = src.shape[0] if args.days <= 0 else min(args.days, src.shape[0])
    dst = np.lib.format.open_memmap(args.out, mode="w+", dtype=np.float16,
                                    shape=(nt,) + src.shape[1:])
    sig_px = sig / DY_KM
    kept = tot = 0
    for lo in range(0, nt, args.block):
        hi = min(lo + args.block, nt)
        blk = np.asarray(src[lo:hi], np.float32)
        for k in range(blk.shape[0]):
            y = blk[k]
            v = np.isfinite(y)
            if not v.any():
                dst[lo + k] = np.float16(np.nan)
                continue
            y0 = np.where(v, y, 0.0)
            vf = v.astype(np.float32)
            out = np.zeros_like(y0)
            den_wide = None
            for w, sp in zip(a, sig_px):
                if sp <= 0:
                    out += w * y0
                    continue
                num = gaussian_filter(y0, sp, mode="nearest")
                den = gaussian_filter(vf, sp, mode="nearest")
                out += w * num / np.maximum(den, 1e-6)
                den_wide = den
            # same trust rule as the low-pass build: the WIDEST kernel must have had
            # real support, and the original mask is re-applied so no unobserved
            # pixel becomes observed
            ok = v & (den_wide > 0.25)
            out = np.where(ok, out, np.nan)
            dst[lo + k] = out.astype(np.float16)
            kept += int(np.isfinite(out).sum())
            tot += int(v.sum())
        print(f"  {hi}/{nt}", flush=True)
    dst.flush()
    print(f"\nwrote {args.out}")
    print(f"observed pixels retained: {100*kept/max(tot,1):.3f}%")

    t = nt // 2
    b = np.asarray(np.load(args.out, mmap_mode="r")[t], np.float32)
    aa = np.asarray(src[t], np.float32)
    m = np.isfinite(aa) & np.isfinite(b)
    print(f"day {t}: std raw {np.std(aa[m]):.5f} -> whitened {np.std(b[m]):.5f} "
          f"({100*(1-np.std(b[m])/np.std(aa[m])):.2f}% of amplitude removed)")


if __name__ == "__main__":
    main()
