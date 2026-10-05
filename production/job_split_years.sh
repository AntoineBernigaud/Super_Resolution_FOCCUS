#!/bin/bash
#SBATCH --job-name=sr_split_years
#SBATCH --output=logs/sr_split_years.o%A_%a
#SBATCH --error=logs/sr_split_years.e%A_%a
# account: export SBATCH_ACCOUNT=project_XXXXXXXXX (not hard-coded here)
#SBATCH --time=02:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G
#SBATCH --partition=small
#SBATCH --array=0-33%8          # SLURM caps array indices at 1001, so these are OFFSETS from FIRST_YEAR
#
# Cut SR_duacs_total.nc into one file per calendar year, for publication: Zenodo
# takes 50 GB per record and the whole product is 115 GB, so it goes up as 34
# yearly files of ~3.4 GB that the validation scripts read interchangeably.
#
#   sbatch production/job_split_years.sh                  # all 34 years
#   sbatch --array=32 production/job_split_years.sh        # one year (1993 + 32 = 2025)
#
# Each task is an independent raw copy (no re-encoding), so a failed year can be
# re-run on its own; extract_period.py refuses to overwrite an existing output.
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$REPO/env.sh" ] && [ "$REPO" != "/" ]; do REPO="$(dirname "$REPO")"; done
[ -f "$REPO/env.sh" ] || { echo "cannot find repo root (no env.sh above $SLURM_SUBMIT_DIR)"; exit 1; }
cd "$REPO" || exit 1
mkdir -p logs product_years
source env.sh
FIRST_YEAR=1993
Y=$((FIRST_YEAR + SLURM_ARRAY_TASK_ID))
echo "task $SLURM_ARRAY_TASK_ID -> year $Y"

srun python production/extract_period.py --src SR_duacs_total.nc \
     --start "$Y-01-01" --end "$Y-12-31" --out "product_years/SR_duacs_total_$Y.nc"
