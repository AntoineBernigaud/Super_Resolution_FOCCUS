"""Turn days of the published dataset into an archive the diagnostics can read.

`make_archive.py` samples the model and writes pred_<date>.npz; every diagnostic in
validation/ reads that format.  If you downloaded SR_duacs_total.nc instead of
running the model, this writes the same npz files straight from it -- no GPU, no
checkpoints -- so the whole validation suite works on the published product:

    python validation/archive_from_dataset.py --dataset SR_duacs_total.nc \\
           --split test --days 40 --out archive_wh13
    sbatch validation/job_validate.sh 3.3

The members in the dataset are ALREADY inflated (lambda 3.3 above 200 km), so do not
inflate them again: run the diagnostics at lambda 1.0, and read the lambda in
`inflation` in the file's global attributes.

Needs `patch_index.npz` and `norm_stats.npz` (training/job_index.sh), because the
archive format stores normalised anomalies and indexes days by their position in the
SWOT record -- which is also what every diagnostic needs to find the SWOT truth.  A
date outside that record has no truth to be scored against and is skipped.
"""
import argparse
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

import config as C
from data import load_index, load_stats


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset", help="merged SR_duacs_total.nc")
    g.add_argument("--product", help="directory of per-day files from produce_sr.py")
    ap.add_argument("--split", default="test", choices=list(C.SPLITS) + ["all"])
    ap.add_argument("--days", type=int, default=40)
    ap.add_argument("--dates", nargs="*", default=None)
    ap.add_argument("--min-cov", type=float, default=0.15)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mean, std = load_stats()
    idx = load_index()
    rec = idx["dates"].astype("datetime64[D]")
    cov = idx["day_cov"]
    t_of = {str(d): i for i, d in enumerate(rec)}

    if args.dates:
        want = [np.datetime64(d) for d in args.dates]
    else:
        lo, hi = ((rec.min(), rec.max()) if args.split == "all"
                  else tuple(np.datetime64(d) for d in C.SPLITS[args.split]))
        ok = np.nonzero((rec >= lo) & (rec <= hi) & (cov > args.min_cov))[0]
        if len(ok) == 0:
            raise SystemExit(f"no day of '{args.split}' has coverage > {args.min_cov}")
        # spread evenly over the period, exactly as make_archive.py picks its days
        pick = np.unique(ok[np.linspace(0, len(ok) - 1,
                                        min(args.days, len(ok))).round().astype(int)])
        want = list(rec[pick])

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    src = (Dataset(args.dataset) if args.dataset else None)
    if src is not None:
        src.set_auto_mask(True)
        dsdays = (np.datetime64("1950-01-01")
                  + np.asarray(src["time"][:], np.int64).astype("timedelta64[D]"))
        where = {str(d.astype("datetime64[D]")): i for i, d in enumerate(dsdays)}

    n = skipped = 0
    for d in want:
        key = str(d)
        if key not in t_of:
            print(f"  {key}: outside the SWOT record, no truth to score -- skipped")
            skipped += 1
            continue
        if src is not None:
            i = where.get(key)
            if i is None:
                print(f"  {key}: not in {args.dataset} -- skipped"); skipped += 1; continue
            ens = src["sla"][i].filled(np.nan)
            mu = src["sla_mu"][i].filled(np.nan)
        else:
            f = Path(args.product, key[:4], f"sr_nordic_sla_{key.replace('-', '')}.nc")
            if not f.exists():
                print(f"  {key}: {f} missing -- skipped"); skipped += 1; continue
            with Dataset(f) as ds:
                ds.set_auto_mask(True)
                ens = ds["sla"][0].filled(np.nan)
                mu = ds["sla_mu"][0].filled(np.nan)
        # the archive convention: normalised anomalies, land carried as 0 (the
        # diagnostics mask to finite SWOT anyway, which excludes land)
        enc = lambda a: np.nan_to_num((a - mean) / std).astype(np.float16)
        t = t_of[key]
        np.savez(out / f"pred_{key}.npz", ens=enc(ens), mu=enc(mu),
                 t=int(t), date=key, cov=float(cov[t]))
        n += 1
        print(f"  [{n}/{len(want)}] {key} -> pred_{key}.npz", flush=True)
    if src is not None:
        src.close()
    print(f"wrote {n} day(s) to {out}" + (f", skipped {skipped}" if skipped else ""))


if __name__ == "__main__":
    main()
