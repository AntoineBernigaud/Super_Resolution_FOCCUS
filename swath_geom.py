"""Swath geometry lookup: for any date, which SWOT passes crossed and where.

The index is built by scanning the SWOT L3 files that build_dataset.py downloads
(`downloads/Science`, `downloads_ext/Science`), whose names carry cycle, pass and
date: SWOT_L3_LR_SSH_Expert_<cycle>_<pass>_<YYYYMMDD>T...  Set $SR_SWOT_DIRS (colon
separated) to look elsewhere.  If the older
`swath_geometry/sr_dataset/swath_geometry_index.csv` exists it is used instead: it
lists every (date, pass) of the record and names which file holds that pass's
geometry, so it also covers dates whose own files are not on disk.

Geometry repeats from cycle to cycle, so a date whose own files were not downloaded
can borrow another date's passes; `validation/swath_fullperiod.py` does exactly that
for the pre-SWOT years.
"""
import csv
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

INDEX = Path("swath_geometry/sr_dataset/swath_geometry_index.csv")   # older layout
ROOT = Path("swath_geometry")
FNAME_RE = re.compile(r"SWOT_L3_LR_SSH_Expert_(\d{3})_(\d{3})_(\d{8})T")


def swot_dirs():
    env = os.environ.get("SR_SWOT_DIRS")
    if env:
        return [Path(p) for p in env.split(":") if p]
    return [Path("downloads/Science"), Path("downloads_ext/Science"), ROOT]


@lru_cache(maxsize=1)
def index():
    """{'YYYY-MM-DD': [(pass, file, file), ...]}, from the downloaded L3 files.

    The third element is the same file: in the old CSV the geometry and the SSH
    could live in different files, and callers still unpack three.
    """
    by_date = {}
    if INDEX.exists():
        # Preferred when present: the CSV lists every (date, pass) in the record and
        # names which file carries that pass's geometry, so it covers dates whose own
        # files were never downloaded.  A bare filename scan only covers the dates
        # actually on disk.
        with open(INDEX, newline="") as fh:
            for r in csv.DictReader(fh):
                d = r["date"].strip()
                key = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
                by_date.setdefault(key, []).append(
                    (int(r["pass"]), r["geometry_file"].strip(), r["swot_file"].strip()))
        return by_date
    for d in swot_dirs():
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("*.nc")):
            m = FNAME_RE.search(f.name)
            if not m:
                continue
            day = m.group(3)
            key = f"{day[:4]}-{day[4:6]}-{day[6:8]}"
            row = (int(m.group(2)), str(f), str(f))
            if row not in by_date.setdefault(key, []):
                by_date[key].append(row)
    return by_date


@lru_cache(maxsize=1)
def _paths():
    """{name or path: path} -- callers pass whichever the index gave them."""
    out = {}
    for d in swot_dirs():
        if d.is_dir():
            for p in d.rglob("*.nc"):
                out[p.name] = p
                out[str(p)] = p
    return out


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
