#!/bin/bash
#SBATCH --job-name=sr_prodtest
#SBATCH --output=logs/sr_prodtest.o%j
#SBATCH --error=logs/sr_prodtest.e%j
#SBATCH --account=project_465002856
#SBATCH --time=01:30:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=7 --mem=56G
#SBATCH --partition=small-g --gpus=1
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$REPO/env.sh" ] && [ "$REPO" != "/" ]; do REPO="$(dirname "$REPO")"; done
cd "$REPO" || exit 1
mkdir -p logs
source env.sh
[ -f DUACS_full.nc ] || { echo "DUACS_full.nc missing: place the DUACS L4 sla record at the repo root"; exit 1; }
rm -rf product_test
for d in 2025-07-21 1995-06-01 2020-06-01 2024-01-15; do
  srun python production/produce_sr.py --start $d --end $d --out product_test || exit 1
done
srun python production/check_product.py product_test
