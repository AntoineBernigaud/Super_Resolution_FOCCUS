#!/bin/bash
#SBATCH --job-name=sr_chkpaths
#SBATCH --output=logs/sr_chkpaths.o%j
#SBATCH --error=logs/sr_chkpaths.e%j
#SBATCH --account=project_465002856
#SBATCH --time=00:20:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=7 --mem=56G
#SBATCH --partition=small-g --gpus=1
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
echo "########## path + import check ##########"
srun python training/_chk_paths.py || exit 1
echo "########## smoke test ##########"
srun python training/smoke_test.py
