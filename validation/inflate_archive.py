"""Write a scale-selectively inflated copy of an archive, so the whole diagnostic
suite can be run on the CALIBRATED ensemble.

`diag_inflate.py` and `diag_long_psd.py` apply the inflation internally and score it,
but they are the only two scripts that know about it -- every other diagnostic reads
the archive raw, so every figure outside those two directories describes the
UNINFLATED ensemble.  Rather than teach fifteen scripts about lam, this materialises
the inflated ensemble once, exactly as `diag_inflate` scores it, and the suite is then
pointed at the new archive with no other change.

    ens' = bar + (ens - bar) + (lam - 1) * LowPass_{>above_km}(ens - bar)

`inflate_large` is imported from `diag_long_psd` rather than reimplemented, so the two
cannot drift apart.  Defaults are the settled configuration: `--above-km 200
--lam 5.0`, which takes CRPS 1.607 -> 1.491 and spread-skill 0.578 -> 1.05 with the
spectrum below 74 km untouched to three decimals.

READ BEFORE USING THE OUTPUT.  Inflation is CALIBRATION, not skill: it cannot make
members explore the direction mu is wrong in, so the width it adds is uninformative by
construction.  Two consequences for the figures made from this archive:

  * the ensemble MEAN is bit-identical to the uninflated one, so ensemble-mean RMSE,
    MAE and every ensemble-mean map are unchanged by design.  If one of them moves,
    something is wrong.
  * a single member is no longer a draw from the model.  The 111-223 km bands
    overshoot to 1.74-2.48x at lam 5 -- unavoidable with a smooth filter, since the
    true deficit is above ~223 km and anything reaching there also lifts bands that
    were already correct.  So EKE, cross-scale transfer and bicoherence measured here
    are NOT the model's; read them from the uninflated archive.  They are produced
    anyway to show what the calibration costs.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np

import config as C
from inflation import DY_KM, inflate_large, lowpass_anomaly

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="archive_wh13")
    ap.add_argument("--out", default="archive_wh13_sinf")
    ap.add_argument("--lam", type=float, default=5.0)
    ap.add_argument("--above-km", type=float, default=200.0,
                    help="0 = scale-blind inflation (multiplies every band; see "
                         "CLAUDE.md for why that is a cancellation trap)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing --out (it is silently stale otherwise, "
                         "which is how mu.npy went wrong)")
    args = ap.parse_args()

    src = Path(args.archive)
    dst = Path(args.out)
    files = sorted(src.glob("pred_*.npz"))
    if not files:
        raise SystemExit(f"no pred_*.npz in {src}")
    if dst.exists() and any(dst.glob("pred_*.npz")) and not args.force:
        raise SystemExit(f"{dst} already holds an archive -- pass --force to replace it")
    dst.mkdir(parents=True, exist_ok=True)

    sig_km = float(np.sqrt(np.log(2.0) * args.above_km ** 2 / (2 * np.pi ** 2)))
    print(f"{src} -> {dst}")
    print(f"  lam {args.lam}  above {args.above_km:g} km "
          f"(sigma {sig_km:.1f} km = {sig_km / DY_KM:.1f} px)  {len(files)} days")

    sd_in = sd_out = 0.0
    for n, f in enumerate(files):
        z = np.load(f)
        ens = np.asarray(z["ens"], np.float32)
        out = inflate_large(ens, args.lam, args.above_km, DY_KM)

        bar_in, bar_out = ens.mean(0), out.mean(0)
        shift = float(np.nanmax(np.abs(bar_out - bar_in)))
        if shift > 1e-4:                       # metres; the mean must not move at all
            raise SystemExit(f"{f.name}: inflation moved the ensemble mean by {shift:g} m")
        # RATIO only.  An absolute spread over the whole canvas is meaningless --
        # ~70% of it is land and never-observed cells, where mu is unconstrained and
        # the members run to several metres, so a canvas mean reads ~25 cm against
        # the 3.3 cm diag_dispersion measures on observed pixels in the eval box.
        # The ratio is region-independent, which is all this check needs.
        s_in = float(np.sqrt(np.nanmean(ens.var(0))))
        s_out = float(np.sqrt(np.nanmean(out.var(0))))
        sd_in += s_in
        sd_out += s_out

        keep = {k: z[k] for k in z.files if k != "ens"}
        np.savez(dst / f.name, ens=out.astype(np.float16), **keep)
        print(f"  [{n + 1}/{len(files)}] {f.name}  spread x{s_out / max(s_in, 1e-12):.3f}",
              flush=True)

    print(f"done.  mean spread ratio x{sd_out / max(sd_in, 1e-12):.3f} "
          f"(canvas-wide; for calibrated numbers read diag_dispersion)")
    print("  ensemble mean unchanged on every day, as it must be")


if __name__ == "__main__":
    main()
