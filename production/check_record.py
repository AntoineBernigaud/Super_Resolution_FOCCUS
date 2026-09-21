"""Integrity check of the finished product: every day present once, every file opens
with the expected structure, and sampled days are numerically sane."""
from pathlib import Path
import numpy as np
from netCDF4 import Dataset

src = Dataset("DUACS_full.nc")
want = np.datetime64("1950-01-01") + src.variables["time"][:].astype(int).astype("timedelta64[D]")
files = sorted(Path("product").rglob("*.nc"))
got, bad = [], []
VARS = {"sla_duacs": (1, 128, 304), "sla_mu": (1, 1024, 912), "sla": (1, 8, 1024, 912),
        "sla_mean": (1, 1024, 912), "quality_flag": (1, 1024, 912)}
for f in files:
    try:
        with Dataset(f) as d:
            for v, shp in VARS.items():
                assert d.variables[v].shape == shp, (v, d.variables[v].shape)
            got.append(np.datetime64("1950-01-01") + np.timedelta64(int(d.variables["time"][0]), "D"))
    except Exception as e:
        bad.append((f.name, repr(e)[:80]))
got = np.array(got, dtype="datetime64[D]")
print(f"files {len(files)}  unreadable/malformed {len(bad)}")
for b in bad[:10]: print("  BAD", b)
missing = np.setdiff1d(want.astype("datetime64[D]"), got)
dup = len(got) - len(np.unique(got))
print(f"expected days {len(want)}  missing {len(missing)}  duplicated {dup}  "
      f"range {got.min()} .. {got.max()}")
if len(missing): print("  first missing:", missing[:10])

print(f"\n{'year':>6}{'spread cm':>11}{'sd member cm':>14}{'sd mu cm':>10}{'ood %':>8}{'flag0 %':>9}")
for y in range(1993, 2027, 3):
    fs = [f for f in files if f.parent.name == str(y)][::30]
    sp, sm, su, ood, f0 = [], [], [], [], []
    for f in fs:
        with Dataset(f) as d:
            s = d.variables["sla"][0].filled(np.nan)
            mu = d.variables["sla_mu"][0].filled(np.nan)
            q = d.variables["quality_flag"][0][:]
            ocean = np.isfinite(mu)
            sp.append(np.nanmean(np.nanstd(s, 0, ddof=1)[ocean]))
            sm.append(np.nanstd(s[0][ocean])); su.append(np.nanstd(mu[ocean]))
            ood.append(bool(q[ocean].max() & 2)); f0.append(np.mean(q[ocean] == 0))
            assert np.all(np.isfinite(s[:, ocean])), f"NaN inside ocean {f.name}"
    if fs:
        print(f"{y:>6}{100*np.mean(sp):>11.2f}{100*np.mean(sm):>14.2f}{100*np.mean(su):>10.2f}"
              f"{100*np.mean(ood):>8.0f}{100*np.mean(f0):>9.1f}")
