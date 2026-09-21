"""One pass over the record to produce everything the training loop needs to know
up front:

  1. patch_index.npz  -- target coverage of every candidate patch on every day, so
     the Dataset can sample only patches that actually sit under a swath.
  2. norm_stats.npz   -- mean/std of ssha and sla over the TRAIN split only.
  3. diagnostics      -- coverage histogram and a spatial count map, to choose the
     coverage threshold from the data instead of guessing it, and to check that
     coverage filtering does not bias sampling toward one part of the box.

Reads ssha one day at a time (3.7 MB/step); never holds the 3.1 GB array.
"""
import numpy as np
from netCDF4 import Dataset as NC4

import config as C


def integral(mask):
    """Summed-area table, so patch counts are O(1) each regardless of patch size."""
    out = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int32)
    np.cumsum(np.cumsum(mask, axis=0, dtype=np.int32), axis=1, out=out[1:, 1:])
    return out


def box_counts(ii, i0, j0, ph, pw):
    """Counts for every (i0, j0) origin pair, vectorised over the outer product."""
    i0 = np.asarray(i0)[:, None]
    j0 = np.asarray(j0)[None, :]
    return (ii[i0 + ph, j0 + pw] - ii[i0, j0 + pw]
            - ii[i0 + ph, j0] + ii[i0, j0])


def main():
    ph_f, pw_f = C.PATCH_F
    ph_c, pw_c = C.PATCH_C
    oi_c, oj_c = C.patch_origins()
    oi_f = [i * C.REFINE_LAT for i in oi_c]
    oj_f = [j * C.REFINE_LON for j in oj_c]
    n_i, n_j = len(oi_c), len(oj_c)
    print(f"patch {ph_f}x{pw_f} fine ({ph_c}x{pw_c} coarse), "
          f"{n_i} x {n_j} = {n_i * n_j} candidates/day")

    ds = NC4(C.NC, "r")
    ssha, sla = ds.variables["ssha"], ds.variables["sla"]
    # Raw reads: we want NaN as NaN, not a masked array.
    ssha.set_auto_mask(False)
    sla.set_auto_mask(False)
    # Decode the time axis with xarray (lazy -- reads only the coordinate); doing it
    # by hand through cftime trips over a calendar mismatch in the file.
    import xarray as xr
    with xr.open_dataset(C.NC) as _t:
        dates = _t.time.values.astype("datetime64[D]")
    nt = len(dates)
    print(f"{nt} time steps, {dates[0]} -> {dates[-1]}")

    train_lo = np.datetime64(C.SPLITS["train"][0])
    train_hi = np.datetime64(C.SPLITS["train"][1])

    cov_y = np.zeros((nt, n_i, n_j), dtype=np.float32)
    cov_x = np.zeros((nt, n_i, n_j), dtype=np.float32)
    day_cov = np.zeros(nt, dtype=np.float32)

    # Train-split accumulators, in float64 to keep the variance stable over ~1e11
    # samples.  "clipped" applies the +-0.5 m contract clip before accumulating.
    acc = {k: dict(n=0, s=0.0, ss=0.0) for k in
           ("ssha", "ssha_clipped", "sla")}
    n_out_of_clip = 0
    y_min, y_max = np.inf, -np.inf

    # Read in time blocks rather than one step at a time: the file is compressed
    # (784 MB on disk for ~4 GB raw), and per-step reads pay the HDF5 chunk
    # decompression over and over.  Blocking cut this pass from ~29 min to a few.
    BLK = 32
    blk_t0, y_blk, x_blk = -1, None, None

    for t in range(nt):
        if t // BLK != blk_t0:
            blk_t0 = t // BLK
            lo, hi = blk_t0 * BLK, min((blk_t0 + 1) * BLK, nt)
            y_blk = ssha[lo:hi]
            x_blk = sla[lo:hi]
        y = y_blk[t % BLK]
        x = x_blk[t % BLK]
        my = np.isfinite(y)
        mx = np.isfinite(x)
        day_cov[t] = my.mean()

        cnt_y = box_counts(integral(my), oi_f, oj_f, ph_f, pw_f)
        cnt_x = box_counts(integral(mx), oi_c, oj_c, ph_c, pw_c)
        cov_y[t] = cnt_y / float(ph_f * pw_f)
        cov_x[t] = cnt_x / float(ph_c * pw_c)

        if train_lo <= dates[t] <= train_hi:
            v = y[my].astype(np.float64)
            if v.size:
                acc["ssha"]["n"] += v.size
                acc["ssha"]["s"] += v.sum()
                acc["ssha"]["ss"] += (v * v).sum()
                y_min, y_max = min(y_min, v.min()), max(y_max, v.max())
                n_out_of_clip += int((np.abs(v) > C.CLIP_M).sum())
                vc = np.clip(v, -C.CLIP_M, C.CLIP_M)
                acc["ssha_clipped"]["n"] += vc.size
                acc["ssha_clipped"]["s"] += vc.sum()
                acc["ssha_clipped"]["ss"] += (vc * vc).sum()
            u = x[mx].astype(np.float64)
            acc["sla"]["n"] += u.size
            acc["sla"]["s"] += u.sum()
            acc["sla"]["ss"] += (u * u).sum()

        if t % 100 == 0:
            print(f"  {t:4d}/{nt}  {dates[t]}  cov {day_cov[t]:.3f}", flush=True)

    ds.close()

    # --- stats ------------------------------------------------------------------
    def finish(a):
        m = a["s"] / a["n"]
        return m, np.sqrt(max(a["ss"] / a["n"] - m * m, 0.0))

    print("\n" + "=" * 62)
    print("TRAIN-SPLIT STATISTICS  (%s .. %s)" % C.SPLITS["train"])
    print("=" * 62)
    stats = {}
    for k, a in acc.items():
        m, s = finish(a)
        stats[f"{k}_mean"], stats[f"{k}_std"] = m, s
        print(f"  {k:<14} n={a['n']:>13,}  mean {m:+.4f}  std {s:.4f}")
    print(f"  ssha range  {y_min:+.3f} .. {y_max:+.3f} m")
    print(f"  |ssha| > {C.CLIP_M} m: {n_out_of_clip:,} "
          f"({100 * n_out_of_clip / acc['ssha']['n']:.4f}%)")
    print("  contract reference: ssha +0.1142/0.0661, sla +0.1094/0.0590 "
          "(full record, unclipped)")

    # Both fields share a scale and must use the SAME constants -- a per-field
    # normalisation would remove the offset the network should preserve.
    shared_mean = stats["ssha_clipped_mean"]
    shared_std = stats["ssha_clipped_std"]
    print(f"\n  shared normalisation -> mean {shared_mean:+.6f}  std {shared_std:.6f}")
    np.savez(C.STATS, shared_mean=shared_mean, shared_std=shared_std,
             clip_m=C.CLIP_M, **stats)

    # --- coverage diagnostics ----------------------------------------------------
    print("\n" + "=" * 62)
    print("PATCH TARGET COVERAGE")
    print("=" * 62)
    flat = cov_y.ravel()
    print(f"  {flat.size:,} candidate patches over the record")
    for q in (50, 75, 90, 95, 99):
        print(f"  p{q:<3} {np.percentile(flat, q):.3f}")
    print()
    for thr in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        n = int((flat > thr).sum())
        print(f"  cov > {thr:.1f}  {n:>9,}  ({100 * n / flat.size:5.2f}%)  "
              f"~{n / len(day_cov):6.1f} patches/day")
    print(f"\n  per-day field coverage: mean {day_cov.mean():.4f}  "
          f"min {day_cov.min():.4f}  max {day_cov.max():.4f}")
    print(f"  days with coverage < 0.05: {(day_cov < 0.05).sum()}")

    np.savez_compressed(
        C.PATCH_INDEX, cov_y=cov_y, cov_x=cov_x, day_cov=day_cov,
        dates=dates.astype("datetime64[D]").astype(np.int64),
        oi_c=np.array(oi_c), oj_c=np.array(oj_c),
        patch_c=np.array(C.PATCH_C), patch_f=np.array(C.PATCH_F))
    print(f"\nwrote {C.PATCH_INDEX.name} and {C.STATS.name}")

    _plots(cov_y, day_cov, dates, oi_c, oj_c)


def _plots(cov_y, day_cov, dates, oi_c, oj_c):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].hist(cov_y.ravel(), bins=100)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("patch target coverage")
    ax[0].set_ylabel("count")
    ax[0].set_title("candidate patch coverage")

    ax[1].plot(dates.astype("datetime64[D]"), day_cov, lw=0.7)
    ax[1].set_ylabel("finite fraction")
    ax[1].set_title("per-day field coverage")
    ax[1].tick_params(axis="x", rotation=30)

    # Does coverage filtering bias sampling in space?  Count selected patches per
    # origin over the whole record; the 21-day repeat should even this out.
    sel = (cov_y > 0.5).sum(axis=0)
    im = ax[2].imshow(sel, origin="lower", aspect="auto",
                      extent=[oj_c[0], oj_c[-1], oi_c[0], oi_c[-1]])
    ax[2].set_title("patches with cov > 0.5, per origin")
    ax[2].set_xlabel("coarse lon origin")
    ax[2].set_ylabel("coarse lat origin")
    fig.colorbar(im, ax=ax[2])

    fig.tight_layout()
    out = C.ROOT / "figs" / "patch_coverage.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
