"""Datasets.

Normalisation follows the contract: ssha and sla are the same physical quantity in
metres, so both are normalised with the SAME constants (computed on the train split
only, by build_patch_index.py).  ssha is clipped to +-CLIP_M first -- the extreme
values are genuine L3 artifacts that quality_flag does not catch.

NaN handling:
  sla   NaN (land) -> 0 in normalised units, with a validity mask channel alongside.
  ssha  NaN        -> never filled for the loss.  The finite mask travels with every
                      sample and every loss in this repo is masked by it.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

import config as C


def load_stats():
    s = np.load(C.STATS)
    return float(s["shared_mean"]), float(s["shared_std"])


def load_index():
    return np.load(C.PATCH_INDEX)


def split_mask(dates, split):
    lo, hi = (np.datetime64(d) for d in C.SPLITS[split])
    return (dates >= lo) & (dates <= hi)


class PatchDataset(Dataset):
    """Random-access over (day, patch origin) pairs that pass the coverage filter.

    Coverage filtering is a TRAINING-time device only: it keeps the masked loss on
    patches that actually sit under a swath, which limits how much of each patch's
    receptive field is filled rather than observed.  At inference we tile everywhere.

    with_mu=True additionally returns the frozen stage-1 mean for the patch, read
    from the memmap written by precompute_mu.py.
    """

    def __init__(self, split, cov_thresh=0.5, cov_x_thresh=0.95,
                 with_mu=False, mu_path=None, balance=True, max_lat=None,
                 ssha_cache=None):
        idx = load_index()
        dates = idx["dates"].astype("datetime64[D]")
        cov_y, cov_x = idx["cov_y"], idx["cov_x"]
        self.oi_c, self.oj_c = idx["oi_c"], idx["oj_c"]

        keep = (cov_y > cov_thresh) & (cov_x > cov_x_thresh)
        keep &= split_mask(dates, split)[:, None, None]
        if max_lat is not None:
            centre = C.coarse_lat(self.oi_c + C.PATCH_C[0] / 2.0)
            keep &= (centre <= max_lat)[None, :, None]
        t, i, j = np.nonzero(keep)
        self.items = np.stack([t, i, j], axis=1).astype(np.int32)
        self.cov = cov_y[t, i, j]

        # Coverage filtering is spatially biased: SWOT ground tracks converge near
        # the orbit's turning latitude, so patches around 74-77N clear the threshold
        # roughly four times as often as the rest of the box -- and that band sits
        # OUTSIDE the evaluation box (64-74N) and is the most ice-affected.  Left
        # alone, a quarter of the training signal would come from there.  Weighted
        # sampling flattens the per-origin distribution without discarding data.
        counts = np.zeros((len(self.oi_c), len(self.oj_c)), dtype=np.float64)
        np.add.at(counts, (i, j), 1.0)
        self.weights = (1.0 / counts[i, j]) if balance else np.ones(len(i))
        self.weights /= self.weights.sum()

        centre_lat = C.coarse_lat(self.oi_c[i] + C.PATCH_C[0] / 2.0)
        frac_north = float((centre_lat > C.EVAL_LAT[1]).mean())
        w_north = float(self.weights[centre_lat > C.EVAL_LAT[1]].sum())

        self.mean, self.std = load_stats()
        # ssha_cache selects the TARGET.  Training may use a filtered copy; scoring
        # must always use the raw one, so callers pass this explicitly rather than it
        # being a global.
        self.ssha_cache = ssha_cache or C.CACHE_SSHA
        self.with_mu = with_mu
        self.mu_path = mu_path or (C.ROOT / "mu_whitened.npy")
        self._nc = None
        self._mu = None

        print(f"[{split}] {len(self.items):,} patches from "
              f"{len(np.unique(t)):,} days  (cov>{cov_thresh}), "
              f"mean coverage {self.cov.mean():.3f}, "
              f"north of {C.EVAL_LAT[1]}N: {100*frac_north:.1f}% of patches "
              f"-> {100*w_north:.1f}% of sampling weight "
              f"({'balanced' if balance else 'unbalanced'})")

    def __len__(self):
        return len(self.items)

    def _open(self):
        # Opened lazily so each DataLoader worker gets its own handle.  These are
        # the uncompressed float16 memmaps from build_cache.py, not the netCDF:
        # random patch reads out of the compressed file spend all their time in
        # zlib decompressing whole chunks to retrieve 96x96 pixels.
        if self._nc is None:
            self._nc = (np.load(self.ssha_cache, mmap_mode="r"),
                        np.load(C.CACHE_SLA, mmap_mode="r"))
        if self.with_mu and self._mu is None:
            self._mu = np.load(self.mu_path, mmap_mode="r")

    def __getitem__(self, k):
        self._open()
        cache_y, cache_x = self._nc
        t, ic, jc = (int(v) for v in self.items[k])
        i0c, j0c = int(self.oi_c[ic]), int(self.oj_c[jc])
        ph_c, pw_c = C.PATCH_C
        i0f, j0f = i0c * C.REFINE_LAT, j0c * C.REFINE_LON
        ph_f, pw_f = C.PATCH_F

        y = np.asarray(cache_y[t, i0f:i0f + ph_f, j0f:j0f + pw_f], dtype=np.float32)
        x = np.asarray(cache_x[t, i0c:i0c + ph_c, j0c:j0c + pw_c], dtype=np.float32)

        m_y = np.isfinite(y)
        m_x = np.isfinite(x)
        y = np.clip(np.nan_to_num(y, nan=0.0), -C.CLIP_M, C.CLIP_M)
        y = (y - self.mean) / self.std
        y = np.where(m_y, y, 0.0)          # unobserved: never used, masked in loss
        x = np.where(m_x, (np.nan_to_num(x, nan=0.0) - self.mean) / self.std, 0.0)

        out = {
            "x": torch.from_numpy(
                np.stack([x, m_x.astype(np.float32)]).astype(np.float32)),
            "y": torch.from_numpy(y[None].astype(np.float32)),
            "mask": torch.from_numpy(m_y[None].astype(np.float32)),
            "t": t,
        }
        if self.with_mu:
            mu = np.asarray(self._mu[t, i0f:i0f + ph_f, j0f:j0f + pw_f],
                            dtype=np.float32)
            out["mu"] = torch.from_numpy(mu[None])

        return out


class FullFieldDataset(Dataset):
    """Whole 1024x912 days, for stage-1 inference, mu precomputation and scoring."""

    def __init__(self, split=None, ssha_cache=None):
        self.ssha_cache = ssha_cache or C.CACHE_SSHA
        idx = load_index()
        dates = idx["dates"].astype("datetime64[D]")
        self.t = (np.nonzero(split_mask(dates, split))[0] if split
                  else np.arange(len(dates)))
        self.dates = dates
        self.mean, self.std = load_stats()
        self._nc = None

    def __len__(self):
        return len(self.t)

    def _open(self):
        if self._nc is None:
            self._nc = (np.load(self.ssha_cache, mmap_mode="r"),
                        np.load(C.CACHE_SLA, mmap_mode="r"))

    def __getitem__(self, k):
        self._open()
        cache_y, cache_x = self._nc
        t = int(self.t[k])
        y = np.asarray(cache_y[t], dtype=np.float32)
        x = np.asarray(cache_x[t], dtype=np.float32)
        m_y, m_x = np.isfinite(y), np.isfinite(x)
        yc = np.clip(np.nan_to_num(y, nan=0.0), -C.CLIP_M, C.CLIP_M)
        yc = np.where(m_y, (yc - self.mean) / self.std, 0.0)
        xc = np.where(m_x, (np.nan_to_num(x, nan=0.0) - self.mean) / self.std, 0.0)
        return {
            "x": torch.from_numpy(
                np.stack([xc, m_x.astype(np.float32)]).astype(np.float32)),
            "y": torch.from_numpy(yc[None].astype(np.float32)),
            "mask": torch.from_numpy(m_y[None].astype(np.float32)),
            "land": torch.from_numpy(m_x[None].astype(np.float32)),
            "t": t,
        }


def eval_box_slices(nc_path=None):
    """Index slices of the fine grid for the scoring box (lon -5..20, lat 64..74)."""
    import xarray as xr
    with xr.open_dataset(nc_path or C.NC) as ds:
        lat, lon = ds.lat.values, ds.lon.values
    i = np.nonzero((lat >= C.EVAL_LAT[0]) & (lat <= C.EVAL_LAT[1]))[0]
    j = np.nonzero((lon >= C.EVAL_LON[0]) & (lon <= C.EVAL_LON[1]))[0]
    return slice(i[0], i[-1] + 1), slice(j[0], j[-1] + 1)
