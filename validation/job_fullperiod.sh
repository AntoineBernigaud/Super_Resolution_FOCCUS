#!/bin/bash
#SBATCH --job-name=sr_fullper
#SBATCH --output=logs/sr_fullper_%a.o%j
#SBATCH --error=logs/sr_fullper_%a.e%j
#SBATCH --account=project_465002856
#SBATCH --time=04:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G
#SBATCH --partition=small --gpus=0
#SBATCH --array=0-34
# PSD + cross-scale transfer over the whole super-resolved record.  Task 34 is SWOT
# over its whole record; tasks 0-33 are the model fields for 1993..2026, one year
# each.  When all 35 are done:  sbatch validation/job_fullperiod_combine.sh
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$REPO/env.sh" ] && [ "$REPO" != "/" ]; do REPO="$(dirname "$REPO")"; done
cd "$REPO" || exit 1
mkdir -p logs
source env.sh
if [ "$SLURM_ARRAY_TASK_ID" -eq 34 ]; then
  srun python validation/swath_fullperiod.py --truth "$@"
else
  srun python validation/swath_fullperiod.py --year $((1993 + SLURM_ARRAY_TASK_ID)) "$@"
fi
