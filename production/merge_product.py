"""Merge product/YYYY/sr_nordic_sla_YYYYMMDD.nc into one file, SR_duacs_total.nc.

Raw copy: every variable is read and written with auto mask/scale OFF, so the stored
int32 counts (scale_factor 1e-4 m), the uint8 quality flags and the fill values move
across byte for byte -- nothing is decoded to float and re-encoded.  Chunking is one
day per chunk (one member per chunk for `sla`), as in the daily files, so reading a
single day from the merged file costs what it cost before.

While writing, a CRC32 of every variable of every day is recorded; `--verify`
re-reads the merged file and checks every day against them.  Run it before deleting
the daily files.

Written to SR_duacs_total.nc.tmp and renamed only when complete.
"""
import argparse
import json
import os
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

PER_DAY = ("time_coverage_start", "time_coverage_end", "date_created")
BIG = ("sla_duacs", "sla_mu", "sla", "sla_mean", "quality_flag")


def crc(a):
    return zlib.crc32(np.ascontiguousarray(a).tobytes())


def merge(src, out):
    files = sorted(src.rglob("sr_nordic_sla_*.nc"))
    n = len(files)
    print(f"{n} daily files")
    tmp = out.with_suffix(".nc.tmp")
    sums = {}
    with Dataset(files[0]) as f0, Dataset(tmp, "w", format="NETCDF4") as o:
        ref_attrs = {k: f0.getncattr(k) for k in f0.ncattrs() if k not in PER_DAY}
        for name, d in f0.dimensions.items():
            o.createDimension(name, n if name == "time" else len(d))
        for name, v in f0.variables.items():
            v.set_auto_maskandscale(False)
            filt = v.filters() or {}
            chunks = v.chunking()
            kw = dict(zlib=bool(filt.get("zlib")), complevel=filt.get("complevel", 4),
                      shuffle=bool(filt.get("shuffle")))
            if chunks != "contiguous" and chunks is not None:
                kw["chunksizes"] = chunks
            fill = v.getncattr("_FillValue") if "_FillValue" in v.ncattrs() else None
            ov = o.createVariable(name, v.dtype, v.dimensions, fill_value=fill, **kw)
            ov.set_auto_maskandscale(False)
            ov.setncatts({k: v.getncattr(k) for k in v.ncattrs() if k != "_FillValue"})
            if "time" not in v.dimensions:
                ov[:] = v[:]
        o.variables["time"].set_auto_maskandscale(False)
        prev = None
        for i, f in enumerate(files):
            with Dataset(f) as d:
                a = {k: d.getncattr(k) for k in d.ncattrs() if k not in PER_DAY}
                bad = [k for k in set(a) | set(ref_attrs)
                       if str(a.get(k)) != str(ref_attrs.get(k))]
                if bad:
                    raise SystemExit(f"{f.name}: global attributes differ from the first "
                                     f"file: {bad} -- not the same production run")
                t = float(d.variables["time"][0])
                if prev is not None and t <= prev:
                    raise SystemExit(f"{f.name}: time {t} not after {prev}")
                prev = t
                o.variables["time"][i] = t
                day = {}
                for name in BIG:
                    v = d.variables[name]
                    v.set_auto_maskandscale(False)
                    x = v[:]
                    o.variables[name][i] = x[0]
                    day[name] = crc(x[0])
                sums[str(int(t))] = day
            if i % 500 == 0:
                print(f"  {i + 1}/{n}  {f.name}", flush=True)
        t0 = float(o.variables["time"][0]); t1 = float(o.variables["time"][n - 1])
        d0 = np.datetime64("1950-01-01") + np.timedelta64(int(t0), "D")
        d1 = np.datetime64("1950-01-01") + np.timedelta64(int(t1), "D")
        g = dict(ref_attrs)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        g.update(time_coverage_start=f"{d0}T00:00:00Z", time_coverage_end=f"{d1}T23:59:59Z",
                 date_created=now,
                 history=f"{now}: merged from {n} daily files product/YYYY/"
                         f"sr_nordic_sla_YYYYMMDD.nc by production/merge_product.py "
                         f"(raw copy, no re-encoding)")
        o.setncatts(g)
    os.replace(tmp, out)
    out.with_suffix(".crc.json").write_text(json.dumps(sums))
    print(f"wrote {out}  ({out.stat().st_size / 1e9:.1f} GB), {n} days {d0} .. {d1}")


def verify(out):
    sums = json.loads(out.with_suffix(".crc.json").read_text())
    bad = 0
    with Dataset(out) as o:
        for v in o.variables.values():
            v.set_auto_maskandscale(False)
        tt = o.variables["time"][:]
        if len(tt) != len(sums):
            print(f"FAIL: {len(tt)} days in file, {len(sums)} checksums"); bad += 1
        for i, t in enumerate(tt):
            ref = sums.get(str(int(t)))
            if ref is None:
                print(f"FAIL: day {int(t)} has no checksum"); bad += 1; continue
            for name in BIG:
                if crc(o.variables[name][i]) != ref[name]:
                    print(f"FAIL: day {int(t)} {name} differs"); bad += 1
            if i % 1000 == 0:
                print(f"  verified {i + 1}/{len(tt)}", flush=True)
    print("VERIFIED: every variable of every day matches the daily files byte for byte"
          if bad == 0 else f"{bad} FAILURES -- do not delete the daily files")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="product")
    ap.add_argument("--out", default="SR_duacs_total.nc")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    verify(Path(a.out)) if a.verify else merge(Path(a.src), Path(a.out))
