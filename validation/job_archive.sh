#!/bin/bash
#SBATCH --job-name=sr_archive
#SBATCH --output=logs/sr_archive.o%j
#SBATCH --error=logs/sr_archive.e%j
#SBATCH --account=project_465002856
#SBATCH --time=06:00:00
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
# The uninflated ensemble: sigma_max 13 (fitted to THIS network by matching its
# 10-60 km MD/target power to 1.0 -- refit it after any retrain), 32 steps (8 or 16
# leave EKE 14% high through Heun discretisation error), fixed noise per member,
# residual centred.
srun python validation/make_archive.py --sigma-max 13 --steps 32 \
     --split test --days 40 --members 8 --noise-mode fixed \
     --center-residual --out archive_wh13 "$@"
