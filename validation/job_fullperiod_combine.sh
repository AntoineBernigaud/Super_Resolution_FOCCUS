#!/bin/bash
#SBATCH --job-name=sr_fpcomb
#SBATCH --output=logs/sr_fpcomb.o%j
#SBATCH --error=logs/sr_fpcomb.e%j
# account: export SBATCH_ACCOUNT=project_XXXXXXXXX (not hard-coded here)
# partitions below are LUMI names -- see README to map them
#SBATCH --time=00:20:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G
#SBATCH --partition=debug --gpus=0
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$REPO/env.sh" ] && [ "$REPO" != "/" ]; do REPO="$(dirname "$REPO")"; done
cd "$REPO" || exit 1
source env.sh
srun python validation/swath_fullperiod.py --combine "$@"
