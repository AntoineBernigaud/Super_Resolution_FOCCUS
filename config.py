"""Shared constants.  Every script imports from here so the grid geometry, the
patch tiling and the temporal split cannot drift apart between stages.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NC = ROOT / "sr_duacs_to_swot.nc"
PATCH_INDEX = ROOT / "patch_index.npz"
STATS = ROOT / "norm_stats.npz"
CACHE_SSHA = ROOT / "cache_ssha.npy"
CACHE_SLA = ROOT / "cache_sla.npy"

# --- grid (verified against the file by inspect_dataset.py) ---------------------
REFINE_LAT, REFINE_LON = 8, 3
NLAT_C, NLON_C = 128, 304
NLAT_F, NLON_F = NLAT_C * REFINE_LAT, NLON_C * REFINE_LON   # 1024 x 912

# Box edges land on DUACS cell edges, so a coarse cell centre is edge + (k+0.5)*d.
LAT0, LON0 = 62.0, -18.0
DLAT_C = DLON_C = 0.125


def coarse_lat(i):
    return LAT0 + (i + 0.5) * DLAT_C


def coarse_lon(j):
    return LON0 + (j + 0.5) * DLON_C

# --- patch tiling ---------------------------------------------------------------
# Defined in COARSE cells, so every patch is automatically aligned to an 8x3 block
# boundary and no fractional overlap can ever be introduced.
#   12 x 32 coarse -> 96 x 96 fine -> ~167 x 159 km at 69N (near-square in km).
#   96 = 3 * 2^5, so a UNet can downsample 3-4 times without odd shapes.
PATCH_C = (12, 32)
PATCH_F = (PATCH_C[0] * REFINE_LAT, PATCH_C[1] * REFINE_LON)   # (96, 96)
STRIDE_C = (6, 16)          # 50% overlap between candidate patches

# --- temporal split -------------------------------------------------------------
# Contiguous blocks, chronological, with a 21-day buffer at each boundary: one SWOT
# repeat cycle, and long enough for the mesoscale to decorrelate.  Chronological
# rather than interleaved because forward-in-time is the real use case.
SPLITS = {
    "train": ("2023-07-26", "2025-02-28"),
    "val":   ("2025-03-21", "2025-06-30"),
    "test":  ("2025-07-21", "2025-11-17"),
}

# --- target conditioning --------------------------------------------------------
# Contract: ssha reaches -2.6/+3.4 m but 99.969% of values are within +-0.5 m, and
# the outliers are genuine L3 artifacts that quality_flag does not catch.
CLIP_M = 0.5

# --- evaluation box -------------------------------------------------------------
# Training uses the full box; scoring happens only here.
EVAL_LON = (-5.0, 20.0)
EVAL_LAT = (64.0, 74.0)


def patch_origins():
    """Candidate patch origins in coarse-cell units, inclusive of the last aligned
    origin on each axis so the domain edges are not systematically dropped."""
    def axis(n, p, s):
        o = list(range(0, n - p + 1, s))
        if o[-1] != n - p:
            o.append(n - p)
        return o
    return axis(NLAT_C, PATCH_C[0], STRIDE_C[0]), axis(NLON_C, PATCH_C[1], STRIDE_C[1])

# Halo (in coarse cells) around a patch when cropping the coarse increment, so the
# upsample sees the same neighbourhood a full-field pass would.
DC_HALO = 2
