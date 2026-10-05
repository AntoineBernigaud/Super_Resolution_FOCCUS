"""Find and read the published product, however much of it you have.

Zenodo takes 50 GB per record and the whole product is 115 GB, so it is published
one calendar year at a time.  Four layouts are equivalent to every script here:

    SR_duacs_total.nc               the whole record, 1993-01-01 .. 2026-01-16
    SR_duacs_total_<year>.nc        one calendar year, ~3.4 GB  <- Zenodo
    SR_duacs_total_test_period.nc   the 119 test days, ~1.1 GB  <- Hugging Face
    SR_duacs_total_1999.nc + SR_duacs_total_2000.nc + ...   any subset of years

Every file has the same variables, dimensions and attributes; only the length of
the time axis differs, so nothing downstream needs to know which one it got.

Resolution order, first hit wins ($SR_PRODUCT overrides everything):

    $SR_PRODUCT                     a file, a directory or a glob
    <explicit path passed in>       likewise
    ./SR_duacs_total*.nc            the repository root
    ./data/SR_duacs_total*.nc       where download_data.py puts it
    ./product_years/SR_duacs_total*.nc    where job_split_years.sh writes them

A date is served by whichever file holds it; asking for a date nobody holds is an
error that names the files that were searched, not a KeyError.
"""
import os
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parent
SUBDIRS = ["", "data", "product_years"]
EPOCH = np.datetime64("1950-01-01")


def _expand(spec):
    p = Path(spec)
    if any(c in str(spec) for c in "*?["):
        return sorted(Path(p.parent or ".").glob(p.name))
    if p.is_dir():
        return sorted(p.glob("SR_duacs_total*.nc"))
    return [p] if p.exists() else []


def find(spec=None):
    """The product files to read, in time order.  Never empty: raises instead."""
    for cand in (os.environ.get("SR_PRODUCT"), spec):
        if cand:
            files = _expand(cand)
            if not files:
                raise SystemExit(f"no product file matches {cand!r}")
            return files
    for sub in SUBDIRS:
        files = sorted((ROOT / sub).glob("SR_duacs_total*.nc")) if (ROOT / sub).is_dir() else []
        if files:
            return files
    raise SystemExit(
        "no product file found.  Looked for SR_duacs_total*.nc in "
        + ", ".join(str(ROOT / s) if s else str(ROOT) for s in SUBDIRS)
        + ".\nDownload a year from Zenodo, or the test period with "
          "`python download_data.py --what dataset`, or pass the path explicitly.")


class Product:
    """The product as one date-addressed record, spread over any number of files.

    Open lazily, one netCDF handle per file that is actually read, so pointing
    this at 34 yearly files costs the same as pointing it at one.
    """

    def __init__(self, spec=None, quiet=False):
        self.files = find(spec)
        self._open = {}
        self._at = {}                      # 'YYYY-MM-DD' -> (file index, position)
        for fi, f in enumerate(self.files):
            with Dataset(f) as ds:
                t = np.asarray(ds["time"][:], np.int64)
                units = getattr(ds["time"], "units", "days since 1950-01-01")
                ref = (np.datetime64(units.partition(" since ")[2].split()[0])
                       if "since" in units else EPOCH)
            for i, d in enumerate(ref + t.astype("timedelta64[D]")):
                self._at.setdefault(str(d.astype("datetime64[D]")), (fi, i))
        self.dates = np.array(sorted(self._at), dtype="datetime64[D]")
        if not quiet:
            print(f"product: {len(self.dates)} days {self.dates[0]} .. "
                  f"{self.dates[-1]} from {len(self.files)} file(s)"
                  + (f" ({self.files[0].name})" if len(self.files) == 1 else ""))

    # --- reading -----------------------------------------------------------
    def _ds(self, fi):
        if fi not in self._open:
            ds = Dataset(self.files[fi])
            ds.set_auto_mask(True)
            self._open[fi] = ds
        return self._open[fi]

    def __contains__(self, date):
        return str(np.datetime64(date, "D")) in self._at

    def has(self, date):
        return date in self

    def day(self, date, names=("sla", "sla_mu")):
        """{name: array with NaN for land/fill} for one day, decoded to metres."""
        key = str(np.datetime64(date, "D"))
        if key not in self._at:
            raise SystemExit(f"{key} is not in the product you have "
                             f"({self.dates[0]} .. {self.dates[-1]}, "
                             f"{len(self.dates)} days)")
        fi, i = self._at[key]
        ds = self._ds(fi)
        out = {}
        for n in names:
            a = ds[n][i]
            out[n] = a.filled(np.nan) if np.ma.isMaskedArray(a) else a
        return out

    def file_of(self, date):
        """Which file holds that day -- for messages, and for the notebook."""
        return self.files[self._at[str(np.datetime64(date, "D"))][0]]

    def attrs(self):
        with Dataset(self.files[0]) as ds:
            return {k: ds.getncattr(k) for k in ds.ncattrs()}

    def grid(self):
        """(lat, lon, duacs_lat, duacs_lon) -- identical in every file."""
        with Dataset(self.files[0]) as ds:
            return tuple(np.asarray(ds[n][:]) for n in
                         ("latitude", "longitude", "duacs_latitude", "duacs_longitude"))

    def close(self):
        for ds in self._open.values():
            ds.close()
        self._open.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
