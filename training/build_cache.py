"""Decompress the record once into float16 memmaps.

sr_duacs_to_swot.nc is compressed (784 MB on disk, ~4 GB raw).  Every random patch
read forces HDF5 to decompress the whole chunk containing it, so patch-sampled
training would spend nearly all its time in zlib.  Reading straight from an
uncompressed memmap turns that into a page fault.

  cache_ssha.npy  fp16 (828, 1024, 912)  1.55 GB   NaN kept where unobserved
  cache_sla.npy   fp16 (828,  128, 304)  0.06 GB   NaN kept over land

Values stay in raw metres so the normalisation can change without a rebuild.  fp16
spacing near 0.125 m is 6e-5 m = 0.006 cm, three orders below anything we resolve.
NaN survives fp16, so the finite masks are recovered with np.isfinite as before.
"""
import numpy as np
from netCDF4 import Dataset as NC4

import config as C

BLK = 32


def main():
    ds = NC4(C.NC, "r")
    for v in ("ssha", "sla"):
        ds.variables[v].set_auto_mask(False)
    nt = ds.dimensions["time"].size

    out_y = np.lib.format.open_memmap(
        C.CACHE_SSHA, mode="w+", dtype=np.float16, shape=(nt, C.NLAT_F, C.NLON_F))
    out_x = np.lib.format.open_memmap(
        C.CACHE_SLA, mode="w+", dtype=np.float16, shape=(nt, C.NLAT_C, C.NLON_C))

    for lo in range(0, nt, BLK):
        hi = min(lo + BLK, nt)
        out_y[lo:hi] = ds.variables["ssha"][lo:hi].astype(np.float16)
        out_x[lo:hi] = ds.variables["sla"][lo:hi].astype(np.float16)
        print(f"  {hi}/{nt}", flush=True)

    out_y.flush()
    out_x.flush()
    ds.close()

    # Round-trip check against the source, on a day in the middle of the record.
    ds = NC4(C.NC, "r")
    ds.variables["ssha"].set_auto_mask(False)
    t = nt // 2
    ref = ds.variables["ssha"][t]
    got = np.asarray(np.load(C.CACHE_SSHA, mmap_mode="r")[t], dtype=np.float32)
    same_mask = (np.isfinite(ref) == np.isfinite(got)).all()
    d = np.abs(ref[np.isfinite(ref)] - got[np.isfinite(got)]).max()
    print(f"day {t}: mask identical {same_mask}, max |diff| {d:.2e} m")
    assert same_mask and d < 1e-3
    print("cache OK")


if __name__ == "__main__":
    main()
