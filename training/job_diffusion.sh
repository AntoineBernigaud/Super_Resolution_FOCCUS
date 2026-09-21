#!/bin/bash
#SBATCH --job-name=sr_diffusion
#SBATCH --output=logs/sr_diffusion.o%j
#SBATCH --error=logs/sr_diffusion.e%j
#SBATCH --account=project_465002856
#SBATCH --time=12:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=7 --mem=56G
#SBATCH --partition=small-g --gpus=1
# Resolve the repo root from this script's own location, so sbatch works from any
# directory and every relative data path (cache_*.npy, runs/, archive_*) lands at
# the root beside config.py -- which is what config.ROOT resolves to.
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
# 20 epochs, not 60: every stage 2 bottoms at epoch 7-12 and then overfits 15-36%.
srun python training/train_diffusion.py --mu mu_whitened.npy \
     --target-cache cache_ssha_wh.npy --out runs/diffusion_whitened "$@"
