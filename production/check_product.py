"""Does a produced day reproduce the validated archive, and does it read back as CF?"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr

import config as C
from data import load_stats

mean, std = load_stats()
for f in sorted(Path(sys.argv[1]).rglob("*.nc")):
    ds = xr.open_dataset(f)            # decodes scale_factor and _FillValue
    day = str(ds.time.values[0])[:10]
    print(f"\n=== {f.name}  {day}  {f.stat().st_size / 1e6:.1f} MB   "
          f"inflation: {ds.attrs.get('inflation')}")
    sp = np.nanmean(np.nanstd(ds.sla.values[0], axis=0, ddof=1))
    print(f"  mean member spread {sp * 100:.3f} cm")
    for v in ("sla_duacs", "sla_mu", "sla", "sla_mean", "quality_flag"):
        a = ds[v].values
        fin = np.isfinite(a) if a.dtype.kind == "f" else np.ones(a.shape, bool)
        print(f"  {v:13s} {str(a.shape):28s} {a.dtype}  finite {fin.mean():.3f}  "
              f"range [{np.nanmin(a):+.4f}, {np.nanmax(a):+.4f}]")
    q = ds.quality_flag.values[0]
    for bit, name in zip(ds.quality_flag.flag_masks, ds.quality_flag.flag_meanings.split()):
        print(f"  flag {bit} {name:22s} set on {100 * ((q & bit) > 0).mean():5.1f}% of cells")
    mean8 = ds.sla.values[0].mean(0)
    print(f"  sla_mean == mean(sla): max|diff| {np.nanmax(np.abs(mean8 - ds.sla_mean.values[0])):.2e} m"
          f"  (int32 at 1e-4 m rounds each field by <=5e-5)")
    a = Path("archive_wh13_lam3.3") / f"pred_{day}.npz"
    if a.exists():
        z = np.load(a)
        ref = z["ens"].astype(np.float64) * std + mean
        refmu = z["mu"].astype(np.float64) * std + mean
        land = ~np.isfinite(ds.sla_mu.values[0])
        d = np.abs(ds.sla.values[0] - ref)[:, ~land]
        dm = np.abs(ds.sla_mu.values[0] - refmu)[~land]
        print(f"  vs VALIDATED archive_wh13_lam3.3: members max|diff| {d.max():.2e} m, "
              f"median {np.median(d):.1e} m;  mu max|diff| {dm.max():.2e} m")
    ds.close()
