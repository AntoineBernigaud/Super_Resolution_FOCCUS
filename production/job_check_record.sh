#!/bin/bash
#SBATCH --job-name=sr_chkrec
#SBATCH --output=logs/sr_chkrec.o%j
#SBATCH --error=logs/sr_chkrec.e%j
# account: export SBATCH_ACCOUNT=project_XXXXXXXXX (not hard-coded here)
# partitions below are LUMI names -- see README to map them
#SBATCH --time=00:30:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G
#SBATCH --partition=debug --gpus=0
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$REPO/env.sh" ] && [ "$REPO" != "/" ]; do REPO="$(dirname "$REPO")"; done
cd "$REPO" || exit 1
source env.sh
srun python production/check_record.py
