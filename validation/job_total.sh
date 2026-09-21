#!/bin/bash
#SBATCH --job-name=sr_total
#SBATCH --output=logs/sr_total.o%j
#SBATCH --error=logs/sr_total.e%j
#SBATCH --account=project_465002856
#SBATCH --time=03:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --partition=small --gpus=0
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$REPO/env.sh" ] && [ "$REPO" != "/" ]; do REPO="$(dirname "$REPO")"; done
cd "$REPO" || exit 1
mkdir -p logs
source env.sh
LAM=${1:-1.0}
# PSD, EKE PSD and total cross-scale transfer pooled over train + val + test.  At
# lambda != 1 the three archives are first inflated scale-selectively above 200 km,
# exactly as archive_wh13_lam<L> and the production dataset were.
#   sbatch validation/job_total.sh 3.3     # the lambda of the super-resolved dataset
S=""; [ "$LAM" != "1.0" ] && S="_lam$LAM"
for a in archive_wh13_train archive_wh13_val archive_wh13; do
  [ -d "$a" ] || { echo "missing $a -- sample it first with validation/job_archive.sh"; exit 1; }
  if [ -n "$S" ] && [ ! -d "$a$S" ]; then
    srun python validation/inflate_archive.py --archive $a --out $a$S \
         --lam $LAM --above-km 200 || exit 1
  fi
done
srun python validation/swath_splits.py --whole-plot --no-block-repeat \
     --archives train=archive_wh13_train$S val=archive_wh13_val$S test=archive_wh13$S \
     --out validation/plots/lam$LAM/swath
