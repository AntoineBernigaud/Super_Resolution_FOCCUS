"""Cut a date range out of SR_duacs_total.nc into a smaller file.

Raw copy, exactly as merge_product.py builds the whole file: every variable is read
and written with auto mask/scale OFF, so the stored int32 counts (scale_factor 1e-4 m)
and the uint8 flags move across byte for byte -- nothing is decoded to float and
re-encoded, and the output is bit-identical to the same days of the input.

    python production/extract_period.py --split test
    python production/extract_period.py --src sr_duacs_to_swot.nc --split test \
           --out sr_dataset/sr_duacs_to_swot_test_period.nc

Cheap enough to run without a job: the test period is 120 days, ~1.1 GB in and out.
"""
import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

import config as C

def epoch_of(tvar):
    """The reference date of a CF time axis: 'days since YYYY-MM-DD[ ...]'.

    The product counts days since 1950 and the training dataset counts days since
    its own first day, so this is read from the file rather than assumed.
    """
    units = getattr(tvar, "units", "days since 1950-01-01")
    if "since" not in units:
        raise SystemExit(f"cannot read the time units: {units!r}")
    kind, _, ref = units.partition(" since ")
    if kind.strip() != "days":
        raise SystemExit(f"time is in {kind!r}, only 'days since ...' is handled")
    return np.datetime64(ref.strip().split()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="SR_duacs_total.nc")
    ap.add_argument("--split", choices=list(C.SPLITS),
                    help="use the dates of this split from config.py")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.split:
        start, end = (np.datetime64(d) for d in C.SPLITS[args.split])
    elif args.start and args.end:
        start, end = np.datetime64(args.start), np.datetime64(args.end)
    else:
        raise SystemExit("give --split, or both --start and --end")
    out = Path(args.out or f"SR_duacs_total_{args.split or 'period'}.nc")
    if out.exists():
        raise SystemExit(f"{out} exists; remove it or pass a different --out")

    with Dataset(args.src) as src, Dataset(out.with_suffix(".nc.tmp"), "w",
                                           format="NETCDF4") as o:
        for v in src.variables.values():
            v.set_auto_maskandscale(False)
        epoch = epoch_of(src["time"])
        days = epoch + np.asarray(src["time"][:], np.int64).astype("timedelta64[D]")
        sel = np.nonzero((days >= start) & (days <= end))[0]
        if not len(sel):
            raise SystemExit(f"no day of {args.src} falls in {start}..{end}")
        print(f"{len(sel)} days, {days[sel[0]]} .. {days[sel[-1]]}")

        for name, d in src.dimensions.items():
            o.createDimension(name, len(sel) if name == "time" else len(d))
        for name, v in src.variables.items():
            filt = v.filters() or {}
            ch = v.chunking()
            kw = dict(zlib=bool(filt.get("zlib")), complevel=filt.get("complevel", 4),
                      shuffle=bool(filt.get("shuffle")))
            if ch not in ("contiguous", None):
                # a chunk may be deeper than the slice we keep (the training dataset
                # chunks 138 days at a time); netCDF refuses a chunk bigger than its
                # dimension
                kw["chunksizes"] = [min(c, len(o.dimensions[d]))
                                    for c, d in zip(ch, v.dimensions)]
            fill = v.getncattr("_FillValue") if "_FillValue" in v.ncattrs() else None
            ov = o.createVariable(name, v.dtype, v.dimensions, fill_value=fill, **kw)
            ov.set_auto_maskandscale(False)
            ov.setncatts({k: v.getncattr(k) for k in v.ncattrs() if k != "_FillValue"})
            if "time" not in v.dimensions:
                ov[:] = v[:]
        # Copy in BLOCKS, not day by day: the source may be chunked many days deep
        # (the training dataset uses 138), and a per-day read would decompress that
        # whole chunk once per day -- minutes become an hour.
        contiguous = bool(np.all(np.diff(sel) == 1))
        step = 30
        for k0 in range(0, len(sel), step):
            k1 = min(k0 + step, len(sel))
            for name, v in src.variables.items():
                if "time" not in v.dimensions:
                    continue
                if contiguous:
                    o.variables[name][k0:k1] = v[sel[k0]:sel[k1 - 1] + 1]
                else:
                    for k in range(k0, k1):
                        o.variables[name][k] = v[sel[k]]
            print(f"  {k1}/{len(sel)}  {days[sel[k1 - 1]]}", flush=True)

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        g = {k: src.getncattr(k) for k in src.ncattrs()}
        g.update(time_coverage_start=f"{days[sel[0]]}T00:00:00Z",
                 time_coverage_end=f"{days[sel[-1]]}T23:59:59Z", date_created=now,
                 history=f"{now}: {len(sel)} days ({days[sel[0]]}..{days[sel[-1]]}) "
                         f"extracted from {Path(args.src).name} by "
                         f"production/extract_period.py (raw copy, no re-encoding)\n"
                         + str(g.get("history", "")))
        o.setncatts(g)
    os.replace(out.with_suffix(".nc.tmp"), out)
    print(f"wrote {out}  ({out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
