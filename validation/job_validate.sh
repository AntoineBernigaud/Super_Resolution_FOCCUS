#!/bin/bash
#SBATCH --job-name=sr_validate
#SBATCH --output=logs/sr_validate.o%j
#SBATCH --error=logs/sr_validate.e%j
#SBATCH --account=project_465002856
#SBATCH --time=08:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --partition=small --gpus=0
# SLURM copies the batch script to /var/spool/slurmd, so ${BASH_SOURCE[0]} does NOT
# point into the repo -- walk up from the submit directory to the root that holds
# env.sh instead.  Works whether sbatch was run from the repo root, training/ or
# validation/.
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$REPO/env.sh" ] && [ "$REPO" != "/" ]; do REPO="$(dirname "$REPO")"; done
[ -f "$REPO/env.sh" ] || { echo "cannot find repo root (no env.sh above $SLURM_SUBMIT_DIR)"; exit 1; }
cd "$REPO" || exit 1
mkdir -p logs
source env.sh
LAM=${1:?usage: sbatch job_validate.sh <lambda>   e.g. 5.0 | 4.6 | 3.3 | 1.0}
A=archive_wh13
O=validation/plots/lam$LAM
# lambda = 1.0 means the raw ensemble; anything else is scale-SELECTIVE inflation
# above 200 km, materialised into its own archive so every diagnostic below reads a
# consistent ensemble without needing to know about lambda.
if [ "$LAM" != "1.0" ]; then
  A=archive_wh13_lam$LAM
  srun python validation/inflate_archive.py --archive archive_wh13 --out $A \
       --lam $LAM --above-km 200 --force
fi
echo "########## lambda $LAM -> $O   (archive $A) ##########"
# --- probabilistic: read these from the INFLATED archive ---
srun python validation/diag_crps.py    --archives lam$LAM=$A --out $O/crps
srun python validation/diag_rmse.py    --archive $A --out $O/rmse
# --- physical: at lambda > 1 these show the COST of calibration, not the model ---
srun python validation/swath_splits.py    --archives test=$A --out $O/swath
srun python validation/eke_timeseries.py  --archives test=$A --out $O/eke
srun python validation/diag_phase.py      --archives lam$LAM=$A --surrogates 24 \
     --out $O/phase
srun python validation/diag_overlay_bias.py --archive $A --out $O/overlay_bias
srun python validation/diag_coherence.py  --archive $A --out $O/coherence
srun python validation/plot_days_from_archive.py --archive $A --geo --out $O/daily
