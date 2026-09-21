#!/bin/bash
#SBATCH --job-name=sr_merge
#SBATCH --output=logs/sr_merge.o%j
#SBATCH --error=logs/sr_merge.e%j
#SBATCH --account=project_465002856
#SBATCH --time=20:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G
#SBATCH --partition=small --gpus=0
# Merge product/ into SR_duacs_total.nc (~108 GB).  Then: sbatch production/job_merge.sh --verify
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$REPO/env.sh" ] && [ "$REPO" != "/" ]; do REPO="$(dirname "$REPO")"; done
cd "$REPO" || exit 1
mkdir -p logs
source env.sh
if [ "$1" != "--verify" ]; then
  # stripe the ~108 GB file over 8 OSTs; HDF5 truncates on create but Lustre keeps
  # the layout, so it applies to what netCDF writes
  rm -f SR_duacs_total.nc.tmp && lfs setstripe -c 8 -S 16M SR_duacs_total.nc.tmp
fi
srun python production/merge_product.py "$@"
