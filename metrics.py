"""Verification that does not reduce to RMSE.

RMSE is the wrong headline metric for this task.  A sample that puts a realistic
20 km eddy 10 km from where SWOT saw it is penalised twice -- once for the feature
being absent where observed, once for it being present where not -- and so scores
worse than mu, which produces no eddy at all and is penalised once.  Optimising RMSE
selects for blur, which is the opposite of the point.  It stays in the table as a
sanity check; these are the metrics that carry the argument.

  1. PSD + magnitude-squared coherence -> effective resolution in km.
     Separates the two failure modes RMSE conflates: PSD says whether the model has
     the right amount of energy at each scale, coherence says whether it is in the
     right place.  A blurred field fails on PSD; a displaced-but-realistic field
     fails on coherence but passes PSD.  The wavelength at which coherence^2 drops
     through 0.5 is the standard "resolved scale" in the altimetry literature and is
     the headline number: DUACS resolves ~150 km, SWOT ~15-20 km.

  2. CRPS + rank histogram.  A proper score for an ensemble, so a sharp but slightly
     displaced ensemble is treated far more kindly than RMSE treats it, and the rank
     histogram says whether the truth looks like a draw from the ensemble (flat) or
     the spread is wrong (U-shaped = under-dispersed, dome = over-dispersed).

  3. Geostrophic velocity statistics.  What altimetry is actually used for is surface
     currents, which are a gradient of the field, so they are far more sensitive to
     fine scale than SSH itself.  If the model invents plausible SSH texture with the
     wrong gradient statistics, this is where it shows.

  4. Fractions Skill Score at a range of neighbourhood sizes -- the standard
     displacement-tolerant score.  The neighbourhood at which FSS becomes useful is a
     direct estimate of the position error.

All of these run on the sparse target: only observed pixels contribute, and the
spectral estimates use segments where the truth is fully valid.
"""
import numpy as np

import config as C

DY_KM = C.DLAT_C / C.REFINE_LAT * 111.320      # 1.7394 km, constant with latitude
OMEGA = 7.2921e-5
G = 9.81


# --------------------------------------------------------------------------------
# 1. spectra
# --------------------------------------------------------------------------------
def psd_coherence(pred, truth, valid, seg=128, step=None):
    """Welch PSD of both fields and their coherence, along LATITUDE.

    Along latitude the grid spacing is constant (1.7394 km) at every point in the
    box; along longitude it varies by a factor 1.6 between 64N and 74N, which would
    smear the wavelength axis.  Segments are taken where the truth is fully valid, so
    both fields are always compared over identical samples.
    """
    step = step or seg // 2
    win = np.hanning(seg)
    wnorm = (win ** 2).sum()
    nf = seg // 2 + 1
    Pxx = np.zeros(nf)
    Pyy = np.zeros(nf)
    Pxy = np.zeros(nf, dtype=complex)
    n = 0

    H, W = pred.shape
    for j in range(W):
        ok = valid[:, j].astype(bool)
        if ok.sum() < seg:
            continue
        i = 0
        while i + seg <= H:
            if ok[i:i + seg].all():
                a = pred[i:i + seg, j]
                b = truth[i:i + seg, j]
                A = np.fft.rfft((a - a.mean()) * win)
                B = np.fft.rfft((b - b.mean()) * win)
                Pxx += (A * A.conj()).real
                Pyy += (B * B.conj()).real
                Pxy += A * B.conj()
                n += 1
                i += step
            else:
                i += 1
    if n == 0:
        return None

    Pxx, Pyy, Pxy = Pxx / n, Pyy / n, Pxy / n
    scale = 2.0 * DY_KM / wnorm                      # -> variance per cycle/km
    freq = np.fft.rfftfreq(seg, d=DY_KM)             # cycles / km
    with np.errstate(divide="ignore", invalid="ignore"):
        coh2 = np.abs(Pxy) ** 2 / (Pxx * Pyy)
        wavelength = np.where(freq > 0, 1.0 / np.maximum(freq, 1e-12), np.inf)
    return dict(wavelength_km=wavelength, freq=freq, n_segments=n,
                psd_pred=Pxx * scale, psd_truth=Pyy * scale,
                coh2=np.clip(np.nan_to_num(coh2), 0, 1))


def effective_resolution(res, level=0.5):
    """Wavelength at which coherence^2 falls through `level`, scanning from long
    wavelengths down.  This is the scale the prediction can be said to resolve."""
    if res is None:
        return float("nan")
    wl, c = res["wavelength_km"][1:], res["coh2"][1:]   # drop the DC bin
    order = np.argsort(-wl)                            # long -> short
    wl, c = wl[order], c[order]
    below = np.nonzero(c < level)[0]
    if below.size == 0:
        return float(wl[-1])                           # resolved to the Nyquist
    k = below[0]
    if k == 0:
        return float(wl[0])                            # never coherent
    # linear interpolation in log(wavelength) across the crossing
    c0, c1 = c[k - 1], c[k]
    w0, w1 = np.log(wl[k - 1]), np.log(wl[k])
    t = (level - c0) / (c1 - c0) if c1 != c0 else 0.0
    return float(np.exp(w0 + t * (w1 - w0)))


# --------------------------------------------------------------------------------
# 2. ensemble scores
# --------------------------------------------------------------------------------
def crps_ensemble(ens, obs):
    """Fair (unbiased) CRPS estimator for a small ensemble.

        CRPS = 1/m sum|x_i - y|  -  1/(m(m-1)) sum_{i<j} |x_i - x_j|

    The fair form removes the bias that makes a small ensemble look artificially
    sharp; with m=8 the biased estimator would flatter us by ~7%.
    """
    m = ens.shape[0]
    term1 = np.abs(ens - obs[None]).mean(axis=0)
    term2 = np.zeros_like(term1)
    for i in range(m):
        for j in range(i + 1, m):
            term2 += np.abs(ens[i] - ens[j])
    return term1 - term2 / (m * (m - 1))


def rank_histogram(ens, obs):
    """Where the truth falls among the sorted members.  Flat = calibrated,
    U-shaped = ensemble too narrow, dome = too wide."""
    ranks = (ens < obs[None]).sum(axis=0)
    return np.bincount(ranks, minlength=ens.shape[0] + 1)


def spread_skill(ens, obs):
    """Ensemble spread against the error of the ensemble mean.  For a calibrated
    ensemble of m members these match up to sqrt((m+1)/m)."""
    m = ens.shape[0]
    rmse = np.sqrt(((ens.mean(axis=0) - obs) ** 2).mean())
    spread = np.sqrt((ens.var(axis=0, ddof=1)).mean())
    return float(rmse), float(spread), float(spread / rmse * np.sqrt((m + 1) / m))


# --------------------------------------------------------------------------------
# 3. geostrophic velocity
# --------------------------------------------------------------------------------
def geostrophic_speed(eta_m, lat_deg, valid=None):
    """|u_g| = (g/f) |grad eta|, on pixels where the central differences are valid.

    Currents are a derivative of SSH, so they weight fine scales far more heavily
    than the field does -- a model can look plausible in SSH and still be badly wrong
    here.
    """
    f = 2 * OMEGA * np.sin(np.deg2rad(lat_deg))[:, None]
    dy = DY_KM * 1000.0
    dx = (C.DLON_C / C.REFINE_LON * 111_320.0
          * np.cos(np.deg2rad(lat_deg)))[:, None]

    detady = np.full_like(eta_m, np.nan)
    detadx = np.full_like(eta_m, np.nan)
    detady[1:-1] = (eta_m[2:] - eta_m[:-2]) / (2 * dy)
    detadx[:, 1:-1] = (eta_m[:, 2:] - eta_m[:, :-2]) / (2 * dx)

    speed = (G / np.abs(f)) * np.hypot(detadx, detady)
    if valid is not None:
        v = valid.astype(bool)
        ok = np.zeros_like(v)
        ok[1:-1, 1:-1] = (v[2:, 1:-1] & v[:-2, 1:-1]
                          & v[1:-1, 2:] & v[1:-1, :-2] & v[1:-1, 1:-1])
        speed = np.where(ok, speed, np.nan)
    return speed


# --------------------------------------------------------------------------------
# 4. fractions skill score
# --------------------------------------------------------------------------------
def fss(pred, truth, valid, threshold, scales_px=(1, 3, 9, 27, 81)):
    """Fractions Skill Score over square neighbourhoods.

    The displacement-tolerant score: a feature displaced by less than the
    neighbourhood still counts as a hit, so the scale at which FSS climbs towards 1
    estimates the model's position error.  Fractions are computed over valid pixels
    only, so gaps neither count as hits nor as misses.
    """
    from scipy.ndimage import uniform_filter

    v = valid.astype(np.float64)
    bp = ((pred > threshold) & valid).astype(np.float64)
    bt = ((truth > threshold) & valid).astype(np.float64)
    out = {}
    for n in scales_px:
        if n == 1:
            fp, ft, w = bp, bt, v
        else:
            k = dict(size=n, mode="constant", cval=0.0)
            w = uniform_filter(v, **k)
            fp = uniform_filter(bp, **k)
            ft = uniform_filter(bt, **k)
        use = w > 0.5                       # windows at least half observed
        if use.sum() == 0:
            out[n] = float("nan")
            continue
        p = (fp[use] / w[use])
        t = (ft[use] / w[use])
        num = ((p - t) ** 2).mean()
        den = (p ** 2).mean() + (t ** 2).mean()
        out[n] = float(1 - num / den) if den > 0 else float("nan")
    return out


def summarise(sample, mu, truth, valid, lat_deg, std_m, ens=None):
    """Everything above for one day, in physical units.  `sample`, `mu`, `truth` are
    in normalised units; `std_m` converts to metres."""
    v = valid.astype(bool)
    out = {}

    for name, field in (("sample", sample), ("mu", mu)):
        res = psd_coherence(field, truth, v)
        out[f"eff_resolution_km_{name}"] = effective_resolution(res)
        if res is not None:
            out[f"psd_{name}"] = res
            # Energy ratio in the band SWOT resolves but DUACS does not.
            band = (res["wavelength_km"] >= 15) & (res["wavelength_km"] <= 100)
            out[f"psd_ratio_15_100km_{name}"] = float(
                res["psd_pred"][band].sum() / max(res["psd_truth"][band].sum(), 1e-30))

    sp_t = geostrophic_speed(truth * std_m, lat_deg, v)
    for name, field in (("sample", sample), ("mu", mu)):
        sp = geostrophic_speed(field * std_m, lat_deg, v)
        both = np.isfinite(sp) & np.isfinite(sp_t)
        out[f"ug_rms_{name}_cms"] = float(100 * np.sqrt((sp[both] ** 2).mean()))
        out[f"ug_rms_ratio_{name}"] = float(
            np.sqrt((sp[both] ** 2).mean() / max((sp_t[both] ** 2).mean(), 1e-30)))
    both = np.isfinite(sp_t)
    out["ug_rms_truth_cms"] = float(100 * np.sqrt((sp_t[both] ** 2).mean()))

    thr = np.percentile(truth[v], 90)
    out["fss_sample"] = fss(sample, truth, v, thr)
    out["fss_mu"] = fss(mu, truth, v, thr)

    if ens is not None and ens.shape[0] > 1:
        e = ens[:, v] * std_m * 100          # cm
        o = truth[v] * std_m * 100
        out["crps_sample_cm"] = float(crps_ensemble(e, o).mean())
        out["crps_mu_cm"] = float(np.abs(mu[v] * std_m * 100 - o).mean())
        r, s, ratio = spread_skill(e, o)
        out["ensmean_rmse_cm"], out["spread_cm"] = r, s
        out["spread_skill_ratio"] = ratio
        out["rank_histogram"] = rank_histogram(e, o).tolist()
    return out
