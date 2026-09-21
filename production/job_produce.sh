#!/bin/bash
#SBATCH --job-name=sr_produce
#SBATCH --output=logs/sr_produce_%a.o%j
#SBATCH --error=logs/sr_produce_%a.e%j
#SBATCH --account=project_465002856
#SBATCH --time=40:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=7 --mem=56G
#SBATCH --partition=small-g --gpus=1
#SBATCH --array=0-31
# The full 1993-2026 record: 12,069 days, 8 members each.  Measured cost 263.6 s per
# day on one GCD -> ~884 GCD-hours (~442 GPU-hours billed), ~9.7 MB per day ->
# ~118 GB.  32 contiguous shards of ~377 days each, ~28 h per shard.
#
# Restartable: completed days are skipped and a day only appears under its final name
# once fully written, so a timed-out or failed shard is fixed by resubmitting that
# array index:   sbatch --array=<i> production/job_produce.sh
# Every member's initial noise is seeded 4242+member regardless of day, so the shards
# join seamlessly in time.
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
while [ ! -f "$REPO/env.sh" ] && [ "$REPO" != "/" ]; do REPO="$(dirname "$REPO")"; done
cd "$REPO" || exit 1
mkdir -p logs
source env.sh
srun python production/produce_sr.py --out product \
     --nshards 32 --shard "$SLURM_ARRAY_TASK_ID" "$@"
