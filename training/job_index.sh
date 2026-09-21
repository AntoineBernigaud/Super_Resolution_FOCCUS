#!/bin/bash
#SBATCH --job-name=sr_index
#SBATCH --output=logs/sr_index.o%j
#SBATCH --error=logs/sr_index.e%j
#SBATCH --account=project_465002856
#SBATCH --time=00:30:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --partition=debug --gpus=0
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
# patch_index.npz + norm_stats.npz -- inputs to every dataset, must run first
srun python training/build_patch_index.py "$@"
