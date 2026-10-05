"""Turn days of the published dataset into an archive the diagnostics can read.

`make_archive.py` samples the model and writes pred_<date>.npz; every diagnostic in
validation/ reads that format.  If you downloaded SR_duacs_total.nc instead of
running the model, this writes the same npz files straight from it -- no GPU, no
checkpoints -- so the whole validation suite works on the published product:

    python validation/archive_from_dataset.py --out archive_wh13      # finds the file
    python validation/archive_from_dataset.py --dataset SR_duacs_total_2025.nc \\
           --split test --days 40 --out archive_wh13
    sbatch validation/job_validate.sh 3.3

With no --dataset it resolves whatever product you have (see product.py): the whole
record, one yearly SR_duacs_total_<year>.nc from Zenodo, or the test-period file.
The days scored are the product's own days intersected with the SWOT record, so
downloading a single year needs no extra arguments.

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
import product as P
from data import load_index, load_stats


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dataset", help="a product file, directory or glob; omit to "
                                     "resolve it automatically (product.py)")
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

    src = None if args.product else P.Product(args.dataset)

    if args.dates:
        want = [np.datetime64(d) for d in args.dates]
    else:
        lo, hi = ((rec.min(), rec.max()) if args.split == "all"
                  else tuple(np.datetime64(d) for d in C.SPLITS[args.split]))
        have = np.ones(len(rec), bool) if src is None else np.isin(rec, src.dates)
        ok = np.nonzero((rec >= lo) & (rec <= hi) & (cov > args.min_cov) & have)[0]
        if len(ok) == 0:
            # the common case: a yearly file from a year SWOT did not fly in, or
            # one that does not overlap the requested split
            inrec = np.nonzero((cov > args.min_cov) & have)[0]
            if len(inrec):
                raise SystemExit(
                    f"no day of '{args.split}' is in the product you have.  Its days "
                    f"that ARE in the SWOT record run {rec[inrec[0]]} .. "
                    f"{rec[inrec[-1]]}: pass --split all, or --dates.")
            raise SystemExit(
                "none of the product's days are in the SWOT record "
                f"({rec.min()} .. {rec.max()}), so there is no truth to score "
                "against.  Scoring diagnostics need a year from 2023 on; for an "
                "earlier year use validation/swath_fullperiod.py (spectra and "
                "cross-scale transfer, no truth needed) or the notebook.")
        # spread evenly over the period, exactly as make_archive.py picks its days
        pick = np.unique(ok[np.linspace(0, len(ok) - 1,
                                        min(args.days, len(ok))).round().astype(int)])
        want = list(rec[pick])

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    n = skipped = 0
    for d in want:
        key = str(d)
        if key not in t_of:
            print(f"  {key}: outside the SWOT record, no truth to score -- skipped")
            skipped += 1
            continue
        if src is not None:
            if key not in src:
                print(f"  {key}: not in the product -- skipped"); skipped += 1; continue
            day = src.day(key, ("sla", "sla_mu"))
            ens, mu = day["sla"], day["sla_mu"]
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
