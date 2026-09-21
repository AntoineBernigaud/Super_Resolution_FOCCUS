#!/bin/bash
#SBATCH --job-name=sr_verify
#SBATCH --output=logs/sr_verify.o%j
#SBATCH --error=logs/sr_verify.e%j
#SBATCH --account=project_465002856
#SBATCH --time=02:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=7 --mem=56G
#SBATCH --partition=small-g --gpus=1
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$REPO/env.sh" ] && [ "$REPO" != "/" ]; do REPO="$(dirname "$REPO")"; done
cd "$REPO" || exit 1
mkdir -p logs
source env.sh
# Does this tree still produce the archive it produced before?  Sample 3 days and
# compare against archive_wh13.
#
# NOT an exact-equality test, and that was measured rather than assumed.  The archive
# is NOT bit-reproducible across jobs on LUMI: re-running the SAME code gave max|diff|
# 1.953e-03 against its own archive, which is exactly 2^-9, one float16 ULP at the
# largest stored values.  MIOpen selects convolution algorithms per job and env.sh
# hands every job a fresh MIOpen cache (concurrent jobs sharing one corrupt each
# other), so the accumulation order varies and the float16 store differs in its last
# bit.  The tolerance below is 4 ULP at 4 m -- an order of magnitude tighter than any
# real behavioural change, which would differ by order the signal itself (cm), not by
# order the storage precision.
echo "########## imports + smoke ##########"
srun python training/_chk_paths.py || exit 1
srun python training/smoke_test.py  || exit 1
echo "########## re-sample 3 days with the pruned code ##########"
srun python validation/make_archive.py --sigma-max 13 --steps 32 --split test \
     --days 3 --members 8 --noise-mode fixed --out archive_verify || exit 1
echo "########## compare against archive_wh13 ##########"
srun python - <<'PY'
import numpy as np, pathlib
ULP = 2.0 ** -9          # float16 spacing at values in [2, 4) metres
TOL = 4 * ULP
a = sorted(pathlib.Path("archive_verify").glob("pred_*.npz"))
bad = n = 0
for f in a:
    g = pathlib.Path("archive_wh13")/f.name
    if not g.exists():
        # the 3-day and 40-day selections pick different dates; nothing to compare
        print(f"  {f.name}: not in archive_wh13, skipped"); continue
    n += 1
    x, y = np.load(f)["ens"].astype(np.float64), np.load(g)["ens"].astype(np.float64)
    d = float(np.nanmax(np.abs(x - y)))
    ok = d <= TOL
    print(f"  {f.name}  max|diff| {d:.3e}  ({d / ULP:.1f} float16 ULP)  "
          f"{'OK' if ok else 'CHANGED'}")
    bad += not ok
if n == 0:
    print("  no overlapping days -- nothing was verified"); raise SystemExit(1)
raise SystemExit(1 if bad else 0)
PY
