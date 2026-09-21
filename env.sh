# Shared environment for FOCCUS_ssh_SR jobs on LUMI.  Sourced by every job_*.sh.
#
# torch comes from the CSC pytorch module (ROCm build for MI250x); xarray / h5netcdf
# come from the Pangu_env venv on scratch, which the module does not carry.

module use /appl/local/csc/modulefiles
module load pytorch/2.7

SCRATCH=${SCRATCH:-/scratch/project_465002856/bernigaud}
PROJDIR=${PROJDIR:-$SCRATCH/FOCCUS_ssh_SR}

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# scripts live in training/ and validation/ while config.py and the other
# shared modules sit at the repo root, so the root must be importable
export PYTHONPATH=$REPO:$SCRATCH/Pangu_env/lib/python3.11/site-packages:$PYTHONPATH

# Per-job scratch dirs.  MIOpen caches MUST be per-job: concurrent GPU jobs sharing
# one cache dir corrupt each other's kernel database.
mkdir -p "$PROJDIR/tmp"
export MPLCONFIGDIR=$(mktemp -d "$PROJDIR/tmp/mplconfig_XXXXXX")
export MIOPEN_USER_DB_PATH=$(mktemp -d "$PROJDIR/tmp/miopen_db_XXXXXX")
export MIOPEN_CUSTOM_CACHE_DIR=$(mktemp -d "$PROJDIR/tmp/miopen_cache_XXXXXX")

cleanup() {
    rm -rf "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR" "$MPLCONFIGDIR"
}
trap cleanup EXIT
