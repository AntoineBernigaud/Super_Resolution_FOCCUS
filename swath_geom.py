"""Swath geometry lookup: for any date, which passes crossed and where they were.

The index CSV lists every (date, pass, cycle) in the record and names which of the
143 copied L3 files carries that pass's latitude/longitude.  Geometry repeats exactly
from cycle to cycle, so those 143 files describe the sampling on all 828 days -- which
is what lets the spectra and the transfer be computed on the val and test splits, not
just the 23 days whose SSH we happen to hold.

The CSV has CRLF line endings; python's text mode strips them, a naive split() does
not, and every filename then fails to match.
"""
import csv
from functools import lru_cache
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

INDEX = Path("swath_geometry/sr_dataset/swath_geometry_index.csv")
ROOT = Path("swath_geometry")


@lru_cache(maxsize=1)
def index():
    """{'YYYY-MM-DD': [(pass, geometry_path), ...]}"""
    by_date = {}
    with open(INDEX, newline="") as fh:
        for r in csv.DictReader(fh):
            d = r["date"].strip()
            key = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
            by_date.setdefault(key, []).append(
                (int(r["pass"]), r["geometry_file"].strip(),
                 r["swot_file"].strip()))
    return by_date


@lru_cache(maxsize=1)
def _paths():
    return {p.name: p for p in ROOT.rglob("*.nc")}


@lru_cache(maxsize=256)
def geometry(fname):
    """(lat, lon) for a pass, native (num_lines, num_pixels) order, lon in -180..180."""
    with Dataset(_paths()[fname]) as d:
        lat = np.ma.filled(d["latitude"][:].astype(np.float64), np.nan)
        lon = np.ma.filled(d["longitude"][:].astype(np.float64), np.nan)
    return lat, np.where(lon > 180, lon - 360, lon)


def ssha(fname):
    """Native L3 ssha_filtered, where we happen to hold the pass's own data."""
    with Dataset(_paths()[fname]) as d:
        a = np.ma.filled(d["ssha_filtered"][:].astype(np.float64), np.nan)
    a[np.abs(a) > 1e3] = np.nan
    return a


def passes_for(date):
    """[(pass, geometry_file), ...] -- geometry is available for every date."""
    return [(p, g) for p, g, _ in index().get(date, [])]


def native_passes_for(date):
    """[(pass, swot_file), ...] for the passes whose OWN data file we hold.

    Only the 23 days covered by the copied files qualify; everywhere else the
    geometry is known but the SSH is not, and the gridded product is the only truth
    available.
    """
    return [(p, s) for p, _, s in index().get(date, []) if have(s)]


def native_dates():
    return sorted(d for d in index() if native_passes_for(d))


def have(fname):
    return fname in _paths()
