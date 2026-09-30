"""The scoring regions, and their slices on the fine grid.

`global` is the whole grid; results are reported on it (the 74-77N band where the
SWOT ground tracks converge is part of the domain and is not excluded).
"""
import numpy as np

import config as C


REGIONS = {
    "blackbox": (C.EVAL_LAT, C.EVAL_LON),
    "area2": ((66.0, 74.0), (-5.0, 10.0)),
    # the whole grid.  Note this includes the 74-77N band where SWOT ground tracks
    # converge and the sea ice sits -- coverage there is ~4x the rest, so "global"
    # is weighted toward a region the other two boxes deliberately exclude.
    "global": ((C.LAT0, C.LAT0 + C.NLAT_C * C.DLAT_C),
               (C.LON0, C.LON0 + C.NLON_C * C.DLON_C)),
}


def region_slices(lat_rng, lon_rng):
    """Fine-grid slices for a lat/lon box, derived from config -- never hardcoded."""
    lat = C.LAT0 + (np.arange(C.NLAT_F) + 0.5) * C.DLAT_C / C.REFINE_LAT
    lon = C.LON0 + (np.arange(C.NLON_F) + 0.5) * C.DLON_C / C.REFINE_LON
    i = np.nonzero((lat >= lat_rng[0]) & (lat <= lat_rng[1]))[0]
    j = np.nonzero((lon >= lon_rng[0]) & (lon <= lon_rng[1]))[0]
    return slice(i[0], i[-1] + 1), slice(j[0], j[-1] + 1)
