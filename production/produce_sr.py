"""Produce the super-resolved sea level dataset from the DUACS L4 record.

One CF-1.8 NetCDF file per day, laid out like the CMEMS DUACS L4 daily files and
encoded like SWOT L3 (int32, scale_factor 1e-4 m, zlib + shuffle):

    sla_duacs   (time, duacs_latitude, duacs_longitude)   DUACS input, native 1/8 deg
    sla_mu      (time, latitude, longitude)               stage-1 deterministic mean
    sla         (time, realization, latitude, longitude)  8 diffusion members
    sla_mean    (time, latitude, longitude)               mean of the 8 members
    quality_flag(time, latitude, longitude)               bitmask, see QUALITY_BITS

The numerics are those of validation/make_archive.py, which produced every validated
archive: same stage-1 input construction (data.FullFieldDataset), sigma_max 13,
32 Heun steps, the residual centred, and a FIXED initial noise per member seeded
4242 + member.  That seed does not depend on the day, so the record can be split
across any number of parallel jobs and each member stays temporally continuous
across the split -- which is what --noise-mode fixed exists to provide.

The members are INFLATED, scale-selectively: lambda 3.3 above 200 km (inflation.py),
the width at which the rank histogram is calibrated (ends ratio 1.008 on the 40 test
days, against 1.659 raw).  The raw ensemble is under-dispersed because all members
share one deterministic mean.  Two consequences to keep in mind: the added spread is
calibration, not information -- it cannot point in the direction mu is wrong -- and
the smooth low-pass that reaches above 200 km also lifts 111-223 km, which was
measured at 1.7-2.5x at lambda 5 and is smaller, but not separately measured, at 3.3.
The ensemble mean is unchanged by the inflation.

Restartable and parallel-safe: a day is written to a temporary name and renamed only
once complete, and existing days are skipped.
"""
import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import distance_transform_edt

import config as C
import edm
from data import load_stats
from inflation import DY_KM, inflate_large
from nets import DiffusionUNet, baseline_from_ckpt

SCALE = 1e-4                       # metres, as SWOT L3 and CMEMS DUACS L4
FILL = np.int32(-2147483647)       # SWOT L3 _FillValue for int32 heights
TRAIN = (np.datetime64("2023-07-26"), np.datetime64("2025-02-28"))
SWOT_ERA = np.datetime64("2023-07-26")

# Bits are only set where degradation was MEASURED (plot_results_whitened/
# skill_strata and duacs_record); a bit with no evidence behind it is decoration.
QUALITY_BITS = {
    1: ("near_coast", "within 25 km of DUACS land: ensemble CRPS skill over DUACS "
                      "falls from 28% offshore to 16% at 10-25 km and 5% within 10 km"),
    2: ("input_out_of_distribution", "this day's DUACS gradient energy is below the "
                                     "1st percentile of the training period: the model "
                                     "is extrapolating.  Set on 89-100% of days in "
                                     "1993-1999 (input ~0.5x the training median) and "
                                     "60-100% until 2015; 0-7% from 2017"),
    4: ("no_swot_validation", "date before the SWOT era (2023-07-26): no independent "
                              "truth exists to validate this day"),
    8: ("in_training_period", "date inside the training period 2023-07-26..2025-02-28: "
                              "the model was trained on SWOT for this day, so agreement "
                              "with SWOT is not an independent evaluation"),
}
LAM, ABOVE_KM = 3.3, 200.0


def grad_energy(sla):
    """Mean |grad sla|^2 over ocean, m^2/m^2 -- the input statistic the model's texture
    amplitude tracks.  Identical to diag_duacs_record.py, which set the threshold."""
    ok = np.isfinite(sla)
    z = np.where(ok, sla, 0.0).astype(np.float64)
    lat = C.coarse_lat(np.arange(C.NLAT_C))
    dy = C.DLAT_C * 111.32e3
    dx = C.DLON_C * 111.32e3 * np.cos(np.deg2rad(lat))[:, None]
    gy = np.gradient(z, axis=0) / dy
    gx = np.gradient(z, axis=1) / dx
    g_ok = (ok & np.roll(ok, 1, 0) & np.roll(ok, -1, 0)
            & np.roll(ok, 1, 1) & np.roll(ok, -1, 1))
    return float(np.mean((gx ** 2 + gy ** 2)[g_ok]))


def enc(a):
    """float metres -> int32 counts with fill, exactly as SWOT L3 stores heights."""
    out = np.full(a.shape, FILL, np.int32)
    ok = np.isfinite(a)
    out[ok] = np.round(a[ok] / SCALE).astype(np.int32)
    return out


def load_models(baseline, diffusion, dev):
    import torch
    bck = torch.load(baseline, map_location="cpu", weights_only=False)
    base = baseline_from_ckpt(bck).to(dev).eval()
    base.load_state_dict(bck["model"])
    dck = torch.load(diffusion, map_location="cpu", weights_only=False)
    net = DiffusionUNet(tuple(dck["args"]["ch"])).to(dev).eval()
    net.load_state_dict(dck["ema"])
    return base, net, float(dck["r_scale"]), bck, dck


def static_fields():
    lat_f = C.LAT0 + (np.arange(C.NLAT_F) + 0.5) * C.DLAT_C / C.REFINE_LAT
    lon_f = C.LON0 + (np.arange(C.NLON_F) + 0.5) * C.DLON_C / C.REFINE_LON
    lat_c = C.coarse_lat(np.arange(C.NLAT_C))
    lon_c = C.coarse_lon(np.arange(C.NLON_C))
    return lat_f, lon_f, lat_c, lon_c


def coast_distance_km(land_c):
    land_f = np.repeat(np.repeat(land_c, C.REFINE_LAT, 0), C.REFINE_LON, 1)
    lat_f = static_fields()[0]
    dy = C.DLAT_C / C.REFINE_LAT * 111.32
    # anisotropic EDT needs one sampling per axis; use each row's own dx by stretching
    # longitude to the domain-centre spacing -- <10% error across 62-78N at 25 km
    dx = C.DLON_C / C.REFINE_LON * 111.32 * np.cos(np.deg2rad(lat_f.mean()))
    return land_f, distance_transform_edt(~land_f, sampling=(dy, dx))


def write_day(path, day, duacs, mu, ens, land_f, coast_km, meta, ood):
    tmp = path.with_suffix(".nc.tmp")
    lat_f, lon_f, lat_c, lon_c = static_fields()
    mean_ens = ens.mean(0)
    for a in (mu, mean_ens):
        a[land_f] = np.nan
    ens[:, land_f] = np.nan

    flag = np.zeros(land_f.shape, np.uint8)
    flag[coast_km < 25.0] |= 1
    if ood:
        flag |= 2
    if day < SWOT_ERA:
        flag |= 4
    if TRAIN[0] <= day <= TRAIN[1]:
        flag |= 8

    z = dict(zlib=True, complevel=4, shuffle=True)
    with Dataset(tmp, "w", format="NETCDF4") as nc:
        nc.createDimension("time", 1)
        nc.createDimension("realization", ens.shape[0])
        nc.createDimension("latitude", C.NLAT_F)
        nc.createDimension("longitude", C.NLON_F)
        nc.createDimension("duacs_latitude", C.NLAT_C)
        nc.createDimension("duacs_longitude", C.NLON_C)

        t = nc.createVariable("time", "f8", ("time",))
        t.units, t.calendar, t.standard_name, t.long_name, t.axis = (
            "days since 1950-01-01 00:00:00", "gregorian", "time", "Time", "T")
        t[:] = (day - np.datetime64("1950-01-01")).astype(int)

        for name, dim, val, sn, ax in (
                ("latitude", "latitude", lat_f, "latitude", "Y"),
                ("longitude", "longitude", lon_f, "longitude", "X"),
                ("duacs_latitude", "duacs_latitude", lat_c, "latitude", None),
                ("duacs_longitude", "duacs_longitude", lon_c, "longitude", None)):
            v = nc.createVariable(name, "f4", (dim,))
            v.units = "degrees_north" if "lat" in name else "degrees_east"
            v.standard_name = sn
            v.long_name = (("DUACS grid " if name.startswith("duacs") else "")
                           + ("latitude" if "lat" in name else "longitude")
                           + " of cell centre")
            if ax:
                v.axis = ax
            v[:] = val

        r = nc.createVariable("realization", "i1", ("realization",))
        r.standard_name, r.long_name = "realization", "ensemble member number"
        r[:] = np.arange(1, ens.shape[0] + 1)

        def height(name, dims, data, long_name, comment, chunks):
            v = nc.createVariable(name, "i4", dims, fill_value=FILL,
                                  chunksizes=chunks, **z)
            v.set_auto_maskandscale(False)
            v.scale_factor, v.add_offset = SCALE, 0.0
            v.units = "m"
            v.standard_name = "sea_surface_height_above_sea_level"
            v.long_name, v.comment = long_name, comment
            v.coordinates = ("time duacs_latitude duacs_longitude"
                             if "duacs" in dims[1] else "time latitude longitude")
            v[:] = enc(data)[None]

        F = (1, C.NLAT_F, C.NLON_F)
        height("sla_duacs", ("time", "duacs_latitude", "duacs_longitude"), duacs,
               "DUACS L4 sea level anomaly, model input",
               "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D, unmodified",
               (1, C.NLAT_C, C.NLON_C))
        height("sla_mu", ("time", "latitude", "longitude"), mu,
               "super-resolved sea level anomaly, deterministic stage-1 mean",
               "the best deterministic field; smooth below ~50 km by construction",
               F)
        v = nc.createVariable("sla", "i4", ("time", "realization", "latitude",
                              "longitude"), fill_value=FILL,
                              chunksizes=(1, 1, C.NLAT_F, C.NLON_F), **z)
        v.set_auto_maskandscale(False)
        v.scale_factor, v.add_offset, v.units = SCALE, 0.0, "m"
        v.standard_name = "sea_surface_height_above_sea_level"
        v.long_name = "super-resolved sea level anomaly, diffusion ensemble member"
        v.coordinates = "time realization latitude longitude"
        v.comment = ("Each member is a realisation: use ONE member for gradients, "
                     f"currents or spectra.  Inflated scale-selectively, lambda {LAM} "
                     f"above {ABOVE_KM:g} km, to a calibrated rank histogram.")
        v[:] = enc(ens)[None]
        height("sla_mean", ("time", "latitude", "longitude"), mean_ens,
               "super-resolved sea level anomaly, mean of the ensemble members",
               "NOT a realisation: its gradients and spectra are wrong by construction",
               F)

        q = nc.createVariable("quality_flag", "u1", ("time", "latitude", "longitude"),
                              chunksizes=F, **z)
        q.long_name, q.standard_name = "quality flag", "status_flag"
        q.flag_masks = np.array(list(QUALITY_BITS), np.uint8)
        q.flag_meanings = " ".join(v[0] for v in QUALITY_BITS.values())
        q.comment = "; ".join(f"{k} {v[0]}: {v[1]}" for k, v in QUALITY_BITS.items())
        q.coordinates = "time latitude longitude"
        q[:] = flag[None]

        g = meta.copy()
        g.update(time_coverage_start=f"{day}T00:00:00Z",
                 time_coverage_end=f"{day}T23:59:59Z",
                 date_created=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        nc.setncatts(g)
    os.replace(tmp, path)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--duacs", default=str(C.ROOT / "DUACS_full.nc"))
    ap.add_argument("--start", default="1993-01-01")
    ap.add_argument("--end", default="2100-01-01")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default="product")
    ap.add_argument("--baseline", default="runs/baseline_whitened/best.pt")
    ap.add_argument("--diffusion", default="runs/diffusion_whitened/best.pt")
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--sigma-max", type=float, default=13.0)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--tile-batch", type=int, default=64)
    args = ap.parse_args()

    edm.SIGMA_MAX = args.sigma_max
    dev = torch.device("cuda")
    mean, std = load_stats()
    base, net, r_scale, bck, dck = load_models(args.baseline, args.diffusion, dev)

    src = Dataset(args.duacs)
    days = (np.datetime64("1950-01-01")
            + src.variables["time"][:].astype(int).astype("timedelta64[D]"))
    sel = np.nonzero((days >= np.datetime64(args.start))
                     & (days <= np.datetime64(args.end)))[0]
    # contiguous shards: a failed shard is one date range to resubmit
    sel = np.array_split(sel, args.nshards)[args.shard]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"shard {args.shard}/{args.nshards}: {len(sel)} days "
          f"{days[sel[0]] if len(sel) else '-'} .. {days[sel[-1]] if len(sel) else '-'}"
          f"   r_scale {r_scale:.5f}  sigma_max {args.sigma_max}  steps {args.steps}")

    # the out-of-distribution threshold is derived here from the training period of
    # the same file, not pasted in, so it cannot drift from the data it describes
    tr = np.nonzero((days >= TRAIN[0]) & (days <= TRAIN[1]))[0]
    g_p1 = float(np.percentile([grad_energy(np.asarray(src.variables["sla"][i],
                                                          np.float64)) for i in tr], 1))
    print(f"OOD threshold: training-period p1 of |grad sla|^2 = {g_p1:.4e} "
          f"({len(tr)} days)")
    land_c = ~np.isfinite(np.asarray(src.variables["sla"][sel[0]]))
    land_f, coast_km = coast_distance_km(land_c)
    noise = {e: torch.randn((1, 1, C.NLAT_F, C.NLON_F), device=dev,
                            generator=torch.Generator(device=dev).manual_seed(4242 + e))
             for e in range(args.members)}
    meta = {
        "Conventions": "CF-1.8, ACDD-1.3",
        "title": "Super-resolved sea level anomaly, Nordic Seas: DUACS L4 to SWOT "
                 "resolution by diffusion",
        "summary": "Daily sea level anomaly on a 1/64 x 1/24 deg grid (~1.7 km) "
                   "super-resolved from DUACS L4 (1/8 deg) by a two-stage model "
                   "trained against SWOT KaRIn L3 ssha_filtered: a deterministic "
                   "stage-1 mean (sla_mu) and an EDM diffusion ensemble (sla) on "
                   "its residual.",
        "processing_level": "L4",
        "source": "DUACS L4 cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
        "training_target": "SWOT L3 LR SSH Expert v2.0.1 ssha_filtered, gridded and "
                           "whitened against native L3, 2023-07-26..2025-02-28",
        "geospatial_lat_min": float(C.LAT0),
        "geospatial_lat_max": float(C.LAT0 + C.NLAT_C * C.DLAT_C),
        "geospatial_lon_min": float(C.LON0),
        "geospatial_lon_max": float(C.LON0 + C.NLON_C * C.DLON_C),
        "model_stage1": f"BaselineNet {args.baseline} epoch {bck['epoch']}",
        "model_stage2": f"DiffusionUNet {args.diffusion} epoch {dck['epoch']}",
        "sampler": f"EDM Heun ODE, MultiDiffusion 96x96 tiles, sigma_max "
                   f"{args.sigma_max}, {args.steps} steps, residual centred, fixed "
                   f"initial noise per member (seed 4242+member)",
        "inflation": f"scale-selective, lambda {LAM} above {ABOVE_KM:g} km, about the "
                     f"ensemble mean (mean unchanged)",
        "ood_threshold_grad2": g_p1,
        "r_scale_m": r_scale * std,
        "comment": "Members are realisations; the ensemble mean is not.  Spread is "
                   "inflated to a calibrated rank histogram (ends ratio 1.01, "
                   "spread-skill ~0.83 on test days); it is calibration, not "
                   "information, since all members share one deterministic mean.  "
                   "See quality_flag.",
    }

    for n, i in enumerate(sel):
        day = days[i].astype("datetime64[D]")
        path = out / str(day)[:4] / f"sr_nordic_sla_{str(day).replace('-', '')}.nc"
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        sla = np.asarray(src.variables["sla"][i], np.float32)
        m_x = np.isfinite(sla)
        # identical to data.FullFieldDataset.__getitem__ -- the input the model saw
        xc = np.where(m_x, (np.nan_to_num(sla, nan=0.0) - mean) / std, 0.0)
        x = torch.from_numpy(np.stack([xc, m_x.astype(np.float32)])
                             .astype(np.float32))[None].to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            mu = base(x).float()
        ens = []
        for e in range(args.members):
            r = edm.sample_field(net, mu, x, n_steps=args.steps,
                                 tile_batch=args.tile_batch, init_noise=noise[e])
            r = r - r.mean()
            ens.append(((mu + r_scale * r)[0, 0].cpu().numpy() * std + mean))
        # inflate on the FULL canvas before land masking, exactly as the validated
        # archive_wh13_lam3.3 was made; the low-pass must see the same field
        ens = inflate_large(np.stack(ens).astype(np.float64), LAM, ABOVE_KM, DY_KM)
        mu_m = mu[0, 0].cpu().numpy().astype(np.float64) * std + mean
        write_day(path, day, sla.astype(np.float64), mu_m, ens, land_f, coast_km, meta,
                  ood=grad_energy(sla.astype(np.float64)) < g_p1)
        print(f"  [{n + 1}/{len(sel)}] {day} -> {path}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
