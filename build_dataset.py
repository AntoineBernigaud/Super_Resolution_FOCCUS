"""
build_dataset.py - download SWOT + DUACS and build the DUACS->SWOT super-resolution
dataset in one file.

Steps:  passes -> swot -> duacs -> build -> concat
Run    python build_dataset.py all
"""

# ===========================================================================
#  CONFIGURATION - everything you would normally change is here
# ===========================================================================

# Set True to build ONLY the test period instead of the full record. Affects
# everything downstream: which cycles are fetched, the DUACS subset, which days
# are built, and the name of the concatenated file.
TEST_ONLY  = False
TEST_START = "2025-07-21"      # test period
TEST_END   = "2025-11-17"

DATE_START = "2023-07-26"      # full record: first day
DATE_END   = "2025-11-17"      # full record: last day

# Leave as it is to auto select the cycles of the SWOT satellite that fall onto the given region 
# for the given period.
CYCLES = "auto"

LON_RANGE = (-18.0, 20.0)      # training box
LAT_RANGE = ( 62.0, 78.0)

DUACS_RES  = 0.125             # DUACS grid spacing (deg)
REFINE_LAT = 8                 # target grid = DUACS refined 8x in lat, 3x in lon
REFINE_LON = 3                 # -> 1024 x 912 cells, ~1.74 x 1.66 km at 69N

# Which downloads to run for the "all" step (1 = yes, 0 = skip). Useful when one
# source is already on disk. An explicit step ("build_dataset.py swot") always
# runs, whatever these say.
DOWNLOAD_SWOT  = 1
DOWNLOAD_DUACS = 1

# Delete the per-day files once they are safely inside the concatenated file.
# They hold the same data, so keeping both doubles the disk for nothing.
DELETE_DAILY_AFTER_CONCAT = 1

MIN_PASS_AREA = 5.0            # drop SWOT passes overlapping the box by less (deg^2)
MAX_FILL_KM   = 1.5            # swath->grid: cells farther than this from a sample stay NaN
INTERP        = "cubic"        # swath->grid interpolation ("cubic" or "linear")
WORKERS       = 4              # parallel FTP connections

# Mask the Gulf of Bothnia: a separate basin, inside the box only because the box
# is a rectangle. Format: ((lat0, lat1), (lon0, lon1)).
MASK_BOXES = [((62.0, 64.0), (15.0, 20.0))]

SWOT_VAR   = "ssha_filtered"   # SWOT target variable (denoised)
DUACS_VAR  = "sla"             # DUACS input variable
DUACS_DATASET = "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D"

GEOJSON   = "KaRIn_2kms_science_geometries.geojson"   # SWOT pass footprints
SWOT_DIRS = ["downloads/Science", "downloads_ext/Science"]  # read from all; write to the last
DUACS_FILE = None              # set below; the period goes in the name
OUT_DIR    = "sr_dataset"      # -> sr_dataset/daily/*.nc and sr_dataset/sr_duacs_to_swot.nc

FTP_HOST = "ftp-access.aviso.altimetry.fr"
FTP_PATH = "/swot_products/l3_karin_nadir/l3_lr_ssh/v2_0_1/Expert/"

# Variables kept when a downloaded SWOT file is stripped (10.8 MB -> 5.2 MB).
# latitude/longitude are NOT optional: they are ordinary variables, and without
# them the file has no geolocation and cannot be gridded.
KEEP_VARS = ["time", "latitude", "longitude", "cross_track_distance",
             "ssha_filtered", "quality_flag", "sigma0",
             "mss", "ocean_tide", "internal_tide", "i_num_line", "i_num_pixel"]
# Without these a file is useless; the rest are copied when present.
REQUIRED_VARS = ["time", "latitude", "longitude", "ssha_filtered"]

# --- credentials: AVISO and CMEMS -------------------------------------
AVISO_USER = ""
AVISO_PASS = ""
CMEMS_USER = ""
CMEMS_PASS = ""

# ===========================================================================

import argparse, ftplib, glob, json, os, re, sys, threading, time
import numpy as np
from netCDF4 import Dataset

if TEST_ONLY:                                # one switch, applied before anything reads them
    DATE_START, DATE_END = TEST_START, TEST_END

FILL = -1e8                                  # SWOT fill values are large negatives
DLAT = DUACS_RES / REFINE_LAT
DLON = DUACS_RES / REFINE_LON
NLAT = int(round((LAT_RANGE[1] - LAT_RANGE[0]) / DLAT))
NLON = int(round((LON_RANGE[1] - LON_RANGE[0]) / DLON))
DAILY_DIR = os.path.join(OUT_DIR, "daily")
# Period in the filenames: otherwise a TEST_ONLY run would silently reuse (or
# overwrite) the full-record DUACS subset and concatenated file.
_TAG = f"{DATE_START.replace('-','')}_{DATE_END.replace('-','')}"
if DUACS_FILE is None:
    DUACS_FILE = f"duacs_ext/duacs_sla_{_TAG}.nc"
CONCAT_FILE = os.path.join(OUT_DIR, f"sr_duacs_to_swot_{_TAG}.nc")
FNAME_RE = re.compile(r"SWOT_L3_LR_SSH_Expert_(\d{3})_(\d{3})_(\d{8}T\d{6})")

# ftplib.all_errors is already a tuple; nesting it makes Python raise TypeError
# while handling the original error, which kills the process instead of retrying.
FTP_ERRORS = ftplib.all_errors + (KeyError,)
# netCDF4/HDF5 is not thread-safe: concurrent calls segfault with no traceback.
NC_LOCK = threading.Lock()


def grid():
    """Target grid CELL CENTRES. Centres, not endpoints: using linspace endpoints
    would shift the grid half a cell and break the 8x3 block alignment with DUACS."""
    lat = LAT_RANGE[0] + (np.arange(NLAT) + 0.5) * DLAT
    lon = LON_RANGE[0] + (np.arange(NLON) + 0.5) * DLON
    return lat, lon


def duacs_grid():
    """The DUACS cell centres the target grid refines (128 x 304)."""
    ny = int(round((LAT_RANGE[1] - LAT_RANGE[0]) / DUACS_RES))
    nx = int(round((LON_RANGE[1] - LON_RANGE[0]) / DUACS_RES))
    return (LAT_RANGE[0] + (np.arange(ny) + 0.5) * DUACS_RES,
            LON_RANGE[0] + (np.arange(nx) + 0.5) * DUACS_RES)


# --------------------------------------------------------------- 1. passes --

def passes_for_box():
    """SWOT pass numbers whose swath overlaps the box by more than MIN_PASS_AREA."""
    from shapely.geometry import shape, box
    b = box(LON_RANGE[0], LAT_RANGE[0], LON_RANGE[1], LAT_RANGE[1])
    out = set()
    for f in json.load(open(GEOJSON))["features"]:
        g = shape(f["geometry"])
        if g.intersects(b) and g.intersection(b).area > MIN_PASS_AREA:
            out.add(int(f["properties"]["pass_number"]))
    return sorted(out)


def parse_fname(name):
    m = FNAME_RE.search(name)
    if not m:
        return None
    c, p, t = m.groups()
    return {"cycle": int(c), "pass": int(p), "day": t[:8]}


def on_disk():
    """{(pass, cycle): path} across every SWOT directory. Later dirs win, so a
    refetched copy overrides a bad one in an earlier (possibly read-only) dir."""
    have = {}
    for d in SWOT_DIRS:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            i = parse_fname(f) if f.endswith(".nc") else None
            if i:
                have[(i["pass"], i["cycle"])] = os.path.join(d, f)
    return have


# ------------------------------------------------------------- 2. SWOT FTP --

def aviso_credentials():
    """Config values first, then environment, then prompt."""
    user = AVISO_USER or os.environ.get("AVISO_USER")
    pw = AVISO_PASS or os.environ.get("AVISO_PASS")
    if not user:
        user = input("AVISO username: ").strip()
    if not pw:
        from getpass import getpass
        pw = getpass("AVISO password: ")
    return user, pw


def _connect(user, pw, cwd=None):
    ftp = ftplib.FTP(FTP_HOST, timeout=120)
    ftp.login(user, pw)
    if cwd:
        ftp.cwd(cwd)
    return ftp


def strip_file(src, dst):
    """Rewrite src keeping only KEEP_VARS.

    Values are copied RAW with auto mask/scale OFF on BOTH sides. The data is
    int32 + scale_factor; decoding on read and re-encoding on write requantises
    every value, and doing it on only one side destroys the data silently."""
    with NC_LOCK, Dataset(src) as s, Dataset(dst, "w", format="NETCDF4") as d:
        d.setncatts({k: s.getncattr(k) for k in s.ncattrs()})
        missing = [v for v in REQUIRED_VARS if v not in s.variables]
        if missing:
            raise KeyError(f"{os.path.basename(src)} lacks {missing}")
        keep = [v for v in KEEP_VARS if v in s.variables]
        need = set()
        for v in keep:
            need.update(s.variables[v].dimensions)
        for n, dim in s.dimensions.items():
            if n in need:
                d.createDimension(n, None if dim.isunlimited() else len(dim))
        for n in keep:
            v = s.variables[n]
            v.set_auto_maskandscale(False)
            fv = v.getncattr("_FillValue") if "_FillValue" in v.ncattrs() else None
            o = d.createVariable(n, v.dtype, v.dimensions, zlib=True, complevel=4,
                                 fill_value=fv)
            o.set_auto_maskandscale(False)
            o.setncatts({k: v.getncattr(k) for k in v.ncattrs() if k != "_FillValue"})
            o[:] = v[:]


def cycles_for_dates(ftp):
    """Science cycles whose files fall inside DATE_START..DATE_END.

    The FTP has no date index, so each candidate cycle directory is listed and
    its file dates are read from the names. A cycle is ~20.85 days, but cycle
    lengths vary (the first and last are partial), so an arithmetic guess can be
    several days out -- hence a generous candidate window, then an exact check.
    Directories numbered >= 200 are the 2023 CalVal 1-day-repeat orbit, not
    science, and are skipped."""
    import datetime as dt
    d0 = dt.date(2023, 7, 26)                     # cycle 1, science phase start
    a = (dt.date.fromisoformat(DATE_START) - d0).days
    b = (dt.date.fromisoformat(DATE_END) - d0).days
    lo, hi = int(a / 20.85) - 2, int(b / 20.85) + 3       # generous candidates

    ftp.cwd(FTP_PATH)
    avail = sorted(int(m.group(1)) for m in
                   (re.match(r"cycle_(\d+)$", n) for n in ftp.nlst()) if m)
    avail = [c for c in avail if c < 200 and lo <= c <= hi]

    keep = []
    for c in avail:
        try:
            ftp.cwd(f"{FTP_PATH}cycle_{c:03d}")
            days = sorted(f.split("_")[7][:8] for f in ftp.nlst() if f.endswith(".nc"))
        except FTP_ERRORS:
            continue
        if days and days[-1] >= DATE_START.replace("-", "") \
                and days[0] <= DATE_END.replace("-", ""):
            keep.append(c)
    return keep


def download_swot(cycles, passes):
    """Fetch every (cycle, pass) not already on disk, strip it, save it.

    Writes only to the LAST entry of SWOT_DIRS. Downloads land on .part, are
    stripped to .strip, and only then take the final name, so an interrupted run
    never leaves a file the resume check would accept as complete."""
    out_dir = SWOT_DIRS[-1]
    os.makedirs(out_dir, exist_ok=True)
    user, pw = aviso_credentials()

    have = on_disk()
    want = [(c, p) for c in cycles for p in passes if (p, c) not in have]
    D0, D1 = DATE_START.replace("-", ""), DATE_END.replace("-", "")
    print(f"{len(cycles)}x{len(passes)} combinations, {len(want)} to fetch "
          f"(~{len(want) * 11.7 / 1000:.1f} GB)")
    if not want:
        return

    by_cycle = {}
    for c, p in want:
        by_cycle.setdefault(c, []).append(p)
    buckets = [[] for _ in range(max(1, WORKERS))]
    for i, c in enumerate(sorted(by_cycle)):
        buckets[i % len(buckets)].append(c)

    done = [0]
    lock = threading.Lock()
    t0 = time.time()

    def work(cyc_list):
        ftp = _connect(user, pw)
        try:
            for cycle in cyc_list:
                cdir = f"{FTP_PATH}cycle_{cycle:03d}"
                try:
                    ftp.cwd(cdir)
                    listing = ftp.nlst()
                except ftplib.error_perm:
                    continue
                except FTP_ERRORS:
                    ftp = _connect(user, pw, cdir)
                    listing = ftp.nlst()
                for p in sorted(by_cycle[cycle]):
                    cand = [f for f in listing
                            if re.search(rf"_{cycle:03d}_{p:03d}_", f) and f.endswith(".nc")]
                    if not cand:
                        continue
                    name = max(cand)                 # highest version suffix
                    day = parse_fname(name)["day"]    # a cycle can straddle an end
                    if not (D0 <= day <= D1):         # date, so filter per file too
                        continue
                    dest = os.path.join(out_dir, name)
                    tmp, stp = dest + ".part", dest + ".strip"
                    for attempt in range(1, 4):
                        try:
                            with open(tmp, "wb") as fh:
                                ftp.retrbinary(f"RETR {name}", fh.write)
                            strip_file(tmp, stp)
                            os.replace(stp, dest)
                            os.remove(tmp)
                            with lock:
                                done[0] += 1
                                if done[0] % 100 == 0:
                                    el = time.time() - t0
                                    print(f"  {done[0]}/{len(want)}  {el/60:.1f} min  "
                                          f"ETA {(len(want)-done[0])*el/done[0]/60:.0f} min",
                                          flush=True)
                            break
                        except FTP_ERRORS as e:
                            for junk in (tmp, stp):
                                if os.path.exists(junk):
                                    os.remove(junk)
                            if attempt == 3:
                                print(f"    FAILED {name}: {e}", flush=True)
                                break
                            # A timed-out RETR can leave its reply queued on the
                            # control socket, so every later response is off by
                            # one. Rebuild rather than reuse the connection.
                            time.sleep(2 ** attempt)
                            try:
                                ftp.close()
                            except Exception:
                                pass
                            ftp = _connect(user, pw, cdir)
        finally:
            try:
                ftp.close()
            except Exception:
                pass

    threads = [threading.Thread(target=work, args=(b,)) for b in buckets if b]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"downloaded {done[0]} files to {out_dir}")


# --------------------------------------------------------------- 3. DUACS --

def download_duacs():
    """Subset DUACS to the box, padded one cell so interpolation has edge nodes."""
    if os.path.exists(DUACS_FILE):
        print(f"{DUACS_FILE} already present")
        return
    # copernicusmarine picks credentials up from the environment (or its own
    # cached login if these are left blank).
    if CMEMS_USER:
        os.environ["COPERNICUSMARINE_SERVICE_USERNAME"] = CMEMS_USER
    if CMEMS_PASS:
        os.environ["COPERNICUSMARINE_SERVICE_PASSWORD"] = CMEMS_PASS
    import copernicusmarine
    os.makedirs(os.path.dirname(DUACS_FILE), exist_ok=True)
    pad = DUACS_RES
    copernicusmarine.subset(
        dataset_id=DUACS_DATASET, variables=[DUACS_VAR],
        minimum_longitude=LON_RANGE[0] - pad, maximum_longitude=LON_RANGE[1] + pad,
        minimum_latitude=LAT_RANGE[0] - pad, maximum_latitude=LAT_RANGE[1] + pad,
        start_datetime=f"{DATE_START}T00:00:00", end_datetime=f"{DATE_END}T23:59:59",
        output_filename=os.path.basename(DUACS_FILE),
        output_directory=os.path.dirname(DUACS_FILE), overwrite=True)
    print(f"-> {DUACS_FILE}")


def open_duacs():
    """DUACS snapped to the exact 128 x 304 grid, ascending lat, -180..180 lon."""
    import xarray as xr
    ds = xr.open_dataset(DUACS_FILE)
    la = "latitude" if "latitude" in ds.coords else "lat"
    lo = "longitude" if "longitude" in ds.coords else "lon"
    if float(ds[la][0]) > float(ds[la][-1]):
        ds = ds.isel({la: slice(None, None, -1)})
    if float(ds[lo].max()) > 180:
        ds = ds.assign_coords({lo: ((ds[lo].values + 180) % 360) - 180}).sortby(lo)
    dlat, dlon = duacs_grid()
    ds = ds.sel({la: dlat, lo: dlon}, method="nearest")   # nearest, not slice:
    if (ds.sizes[la], ds.sizes[lo]) != (dlat.size, dlon.size):  # the pad would give 130x306
        raise ValueError(f"DUACS subset is {(ds.sizes[la], ds.sizes[lo])}")
    return ds, la, lo


# ------------------------------------------------------- 4. build the days --

def masks(ds, la, lo):
    """(fine, coarse) cells to force NaN.

    DUACS land is carried onto the SWOT target because the two products disagree
    about the coastline by up to a cell, which would leave target cells with no
    input. Block replication is exact here (8x3), not an approximation."""
    dlat, dlon = duacs_grid()
    land = None
    for i in range(0, ds.sizes["time"], 25):          # land mask is static in time
        m = ~np.isfinite(ds[DUACS_VAR].isel(time=i).values)
        land = m if land is None else (land | m)
    fine = np.repeat(np.repeat(land, REFINE_LAT, axis=0), REFINE_LON, axis=1)
    coarse = np.zeros_like(land)
    lat_t, lon_t = grid()
    for (a, b), (c, d) in MASK_BOXES:
        fine[np.ix_((lat_t >= a) & (lat_t < b), (lon_t >= c) & (lon_t < d))] = True
        coarse[np.ix_((dlat >= a) & (dlat < b), (dlon >= c) & (dlon < d))] = True
    return fine, coarse


def load_swath(path):
    with Dataset(path) as d:
        lat = np.asarray(d["latitude"][:], float)
        lon = np.asarray(d["longitude"][:], float)
        val = np.asarray(d[SWOT_VAR][:], float)
    lat = np.where(lat < FILL, np.nan, lat)
    lon = np.where(lon < FILL, np.nan, lon)
    val = np.where(val < FILL, np.nan, val)
    lon = np.where(lon > 180, lon - 360, lon)      # SWOT stores 0..360
    return lat, lon, val


def grid_swath(lat, lon, val, lat_t, lon_t):
    """Interpolate one swath onto the target grid; returns (field, touched)."""
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree
    ok = (np.isfinite(lat) & np.isfinite(lon) & np.isfinite(val) &
          (lat >= LAT_RANGE[0]) & (lat <= LAT_RANGE[1]) &
          (lon >= LON_RANGE[0]) & (lon <= LON_RANGE[1]))
    field = np.full((NLAT, NLON), np.nan, np.float32)
    empty = np.zeros((NLAT, NLON), bool)
    if ok.sum() < 50:
        return field, empty
    la, lo, va = lat[ok], lon[ok], val[ok]
    # Only the sub-window the swath crosses; a half orbit is a narrow diagonal.
    i0 = max(0, int((la.min() - LAT_RANGE[0]) / DLAT) - 1)
    i1 = min(NLAT, int((la.max() - LAT_RANGE[0]) / DLAT) + 2)
    j0 = max(0, int((lo.min() - LON_RANGE[0]) / DLON) - 1)
    j1 = min(NLON, int((lo.max() - LON_RANGE[0]) / DLON) + 2)
    if i1 <= i0 or j1 <= j0:
        return field, empty
    LO, LA = np.meshgrid(lon_t[j0:j1], lat_t[i0:i1])
    z = griddata((la, lo), va, (LA, LO), method=INTERP)
    z = np.clip(z, va.min(), va.max())          # cubic can ring near data edges
    # Kill anything the samples do not support: the ~20 km nadir gap, coastlines
    # and the convex-hull overshoot. Distances in km, else the longitude
    # tolerance is ~3x too generous at 69N and the fill leaks across the gap.
    cs = np.cos(np.radians(la.mean()))
    tree = cKDTree(np.column_stack([lo * 111.32 * cs, la * 111.32]))
    d, _ = tree.query(np.column_stack([LO.ravel() * 111.32 * cs,
                                       LA.ravel() * 111.32]), k=1, workers=-1)
    z = np.where(d.reshape(z.shape) <= MAX_FILL_KM, z, np.nan)
    touched = np.isfinite(z)
    field[i0:i1, j0:j1] = np.where(touched, z, np.nan).astype(np.float32)
    full = np.zeros((NLAT, NLON), bool)
    full[i0:i1, j0:j1] = touched
    return field, full


def swot_files_by_day(passes):
    out = {}
    for (p, c), path in on_disk().items():
        if p in passes:
            out.setdefault(parse_fname(os.path.basename(path))["day"], []).append(path)
    return out


def build(force=False):
    import xarray as xr
    lat_t, lon_t = grid()
    dlat, dlon = duacs_grid()
    ds, la, lo = open_duacs()
    fine, coarse = masks(ds, la, lo)
    days_duacs = {str(t)[:10].replace("-", ""): i for i, t in enumerate(ds.time.values)}
    by_day = swot_files_by_day(set(passes_for_box()))
    days = sorted(set(by_day) & set(days_duacs))
    os.makedirs(DAILY_DIR, exist_ok=True)
    print(f"building {len(days)} day(s) -> {DAILY_DIR}")

    t0 = time.time()
    for k, day in enumerate(days, 1):
        out = os.path.join(DAILY_DIR, f"sr_{day}.nc")
        if os.path.exists(out) and not force:
            continue
        s = np.zeros((NLAT, NLON))
        n = np.zeros((NLAT, NLON), np.int16)
        for path in sorted(by_day[day]):
            try:
                lat, lon, val = load_swath(path)
            except (OSError, KeyError, IndexError):
                continue
            f, t = grid_swath(lat, lon, val, lat_t, lon_t)
            if t.any():                  # average where passes overlap, don't overwrite
                s[t] += f[t]
                n[t] += 1
        ssha = np.where(n > 0, s / np.maximum(n, 1), np.nan).astype(np.float32)
        sla = ds[DUACS_VAR].isel(time=days_duacs[day]).values.astype(np.float32)
        ssha[fine] = np.nan
        sla[coarse] = np.nan
        n[fine] = 0

        xr.Dataset(
            dict(ssha=(("lat", "lon"), ssha, {"units": "m", "role": "target"}),
                 sla=(("duacs_lat", "duacs_lon"), sla, {"units": "m", "role": "input"}),
                 n_pass=(("lat", "lon"), n.astype(np.int8))),
            coords=dict(lat=lat_t.astype(np.float32), lon=lon_t.astype(np.float32),
                        duacs_lat=dlat.astype(np.float32),
                        duacs_lon=dlon.astype(np.float32)),
            attrs=dict(date=day, coverage=float(np.isfinite(ssha).mean()),
                       swot_var=SWOT_VAR, duacs_var=DUACS_VAR, interp=INTERP,
                       max_fill_km=MAX_FILL_KM, masked_boxes=str(MASK_BOXES),
                       note="ssha is NaN where no swath flew; mask the loss to it"),
        ).to_netcdf(out, encoding={v: dict(zlib=True, complevel=4)
                                   for v in ("ssha", "sla", "n_pass")})
        if k % 25 == 0 or k == len(days):
            el = time.time() - t0
            print(f"  [{k}/{len(days)}] {day}  {el/60:.1f} min", flush=True)
    ds.close()


def concat():
    """Stack the daily files into one (time, ...) file. Stays lazy: the full ssha
    array is ~3 GB and materialising it will run the machine out of memory."""
    import xarray as xr
    d0, d1 = DATE_START.replace("-", ""), DATE_END.replace("-", "")
    # Filter by date: DAILY_DIR may hold days from a previous, wider run.
    files = [f for f in sorted(glob.glob(os.path.join(DAILY_DIR, "sr_*.nc")))
             if d0 <= os.path.basename(f)[3:11] <= d1]
    if not files:
        raise FileNotFoundError(f"no daily files in {DAILY_DIR} for {d0}..{d1}")
    print(f"concatenating {len(files)} day(s)")
    ds = xr.open_mfdataset(files, combine="nested", concat_dim="time", parallel=False)
    days = [os.path.basename(f)[3:11] for f in files]
    ds = ds.assign_coords(time=("time", np.array(
        [np.datetime64(f"{d[:4]}-{d[4:6]}-{d[6:8]}") for d in days])))
    ds.to_netcdf(CONCAT_FILE, encoding={v: dict(zlib=True, complevel=4)
                                        for v in ds.data_vars})
    ds.close()
    print(f"-> {CONCAT_FILE} ({os.path.getsize(CONCAT_FILE)/1e9:.2f} GB)")

    if DELETE_DAILY_AFTER_CONCAT:
        # Re-open and check the output really holds every day before removing the
        # only other copy. Deleting is irreversible; a failed write is not.
        with xr.open_dataset(CONCAT_FILE) as chk:
            n = chk.sizes.get("time", 0)
        if n != len(files):
            print(f"NOT deleting dailies: {CONCAT_FILE} has {n} time steps, "
                  f"expected {len(files)}")
            return
        for f in files:
            os.remove(f)
        print(f"deleted {len(files)} daily file(s) from {DAILY_DIR}")


def parse_range(s):
    out = set()
    for part in str(s).split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part.strip():
            out.add(int(part))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", nargs="?", default="all",
                    choices=["all", "passes", "swot", "duacs", "build", "concat"])
    ap.add_argument("--force", action="store_true", help="rebuild existing daily files")
    a = ap.parse_args()

    print(f"period {DATE_START} .. {DATE_END}" + ("   [TEST_ONLY]" if TEST_ONLY else ""))
    print(f"box lon {LON_RANGE} lat {LAT_RANGE}   grid {NLAT}x{NLON}   "
          f"DUACS {duacs_grid()[0].size}x{duacs_grid()[1].size}")
    assert duacs_grid()[0].size * REFINE_LAT == NLAT, "grid/DUACS block alignment broken"

    passes = passes_for_box()
    print(f"{len(passes)} passes cross the box")
    if a.step == "passes":
        print(passes)
        return
    if a.step == "swot" or (a.step == "all" and DOWNLOAD_SWOT):
        if str(CYCLES).strip().lower() == "auto":
            ftp = _connect(*aviso_credentials())
            cycles = cycles_for_dates(ftp)
            ftp.close()
            print(f"cycles covering {DATE_START}..{DATE_END}: "
                  f"{cycles[0]}-{cycles[-1]} ({len(cycles)})" if cycles else "no cycles")
        else:
            cycles = parse_range(CYCLES)
        download_swot(cycles, passes)
    if a.step == "duacs" or (a.step == "all" and DOWNLOAD_DUACS):
        download_duacs()
    if a.step in ("all", "build"):
        build(force=a.force)
    if a.step in ("all", "concat"):
        concat()


if __name__ == "__main__":
    sys.exit(main())
