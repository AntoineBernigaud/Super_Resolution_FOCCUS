"""Does the copied tree resolve everything it needs from the NEW root?"""
import importlib, sys
from pathlib import Path
import config as C
print(f"config.ROOT -> {C.ROOT}")
assert C.ROOT.name == "FOCCUS_ssh_SR", f"ROOT is {C.ROOT}, not the new repo"
need = [C.NC, C.CACHE_SSHA, C.CACHE_SLA, C.PATCH_INDEX, C.STATS,
        C.ROOT/"cache_ssha_wh.npy", C.ROOT/"mu_whitened.npy",
        C.ROOT/"runs/baseline_whitened/best.pt", C.ROOT/"runs/diffusion_whitened/best.pt",
        C.ROOT/"runs/native_vs_collocated/native_vs_collocated.json",
        C.ROOT/"archive_wh13", C.ROOT/"swath_geometry/sr_dataset/swath_geometry_index.csv"]
bad = [p for p in need if not Path(p).exists()]
for p in need:
    print(("  OK   " if Path(p).exists() else "  MISS ") + str(p).replace(str(C.ROOT)+"/",""))
mods = ["config","data","nets","edm","metrics","plotting","swath_geom",
        "swath_transfer","fss_analysis"]
for m in mods:
    importlib.import_module(m)
print(f"imported {len(mods)} shared modules OK")
import swath_geom
print(f"swath_geom.INDEX exists: {Path(swath_geom.INDEX).exists()}  ({swath_geom.INDEX})")
sys.exit(1 if bad else 0)
