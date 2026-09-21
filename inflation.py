"""Scale-selective ensemble inflation, shared by validation and production.

    ens' = bar + (ens - bar) + (lam - 1) * LowPass_{>above_km}(ens - bar)

Widens the members about their own mean only above `above_km`, where the missing
variance actually is.  The ensemble mean is unchanged by construction.  Gaussian
low-pass with half AMPLITUDE at the cutoff.
"""
import numpy as np

import config as C

DY_KM = C.DLAT_C / C.REFINE_LAT * 111.320


def lowpass_anomaly(ens, above_km, dy_km):
    """(bar, anomaly, low-passed anomaly) -- the expensive part, independent of lam.

    Split out because the convolution does NOT depend on the inflation factor, and
    recomputing it inside a lam sweep costs one full-field Gaussian per member per lam
    for nothing.
    """
    from scipy.ndimage import gaussian_filter
    sig_px = float(np.sqrt(np.log(2.0) * above_km ** 2 / (2 * np.pi ** 2))) / dy_km
    bar = ens.mean(0)
    a = ens - bar[None]
    lo = np.stack([gaussian_filter(x, sig_px, mode="nearest") for x in a])
    return bar, a, lo


def inflate_large(ens, lam, above_km, dy_km, pre=None):
    """Widen members about the ensemble mean, but only above `above_km`.

    ens' = bar + (ens - bar) + (lam - 1) * LowPass(ens - bar)

    Gaussian low-pass with half AMPLITUDE at `above_km`, matching the convention in
    build_lowpass_target.py so the cutoff means the same thing across the repo.  The
    member anomaly is a model output and has no gaps, so a plain convolution is right
    here -- no normalised convolution needed.
    """
    if lam == 1.0 or above_km <= 0:
        return ens
    bar, a, lo = pre if pre is not None else lowpass_anomaly(ens, above_km, dy_km)
    return bar[None] + a + (lam - 1.0) * lo
