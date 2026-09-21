#!/bin/bash
#SBATCH --job-name=sr_target
#SBATCH --output=logs/sr_target.o%j
#SBATCH --error=logs/sr_target.e%j
#SBATCH --account=project_465002856
#SBATCH --time=02:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --partition=small --gpus=0
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
# The whitened target: multiply the gridded target's amplitudes by
# H(k)=1/sqrt(P_gridded/P_native) so it carries the native L3 spectrum at every
# scale.  Reads runs/native_vs_collocated/native_vs_collocated.json.
srun python training/build_whitened_target.py "$@"
