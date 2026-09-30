# Environment for every job_*.sh.  Sourced, not executed.
#
# Two blocks: SITE, which you adapt, and generic, which you should not need to.
# Everything is overridable from the outside, so you can keep this file untouched
# and export the variables instead.

# ----------------------------- SITE ------------------------------------------
# How to get python + torch.  On LUMI that is a module; elsewhere it is usually a
# conda env or a venv, or nothing at all if python is already on PATH.
#   SR_MODULEPATH  extra module path            (empty = do not call `module use`)
#   SR_MODULES     modules to load              (empty = do not call `module load`)
#   SR_VENV        venv/conda prefix to prepend (empty = skip)
SR_MODULEPATH=${SR_MODULEPATH-/appl/local/csc/modulefiles}
SR_MODULES=${SR_MODULES-pytorch/2.7}
# the CSC pytorch module has torch but not xarray/netCDF4; this venv adds them
SR_VENV=${SR_VENV-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/Pangu_env"}
SR_PY=${SR_PY-python3.11}          # python version inside SR_VENV's site-packages

# The SLURM account is NOT set in the job scripts: export SBATCH_ACCOUNT once
#   export SBATCH_ACCOUNT=project_XXXXXXXXX
# The partitions in the job scripts (small-g / small / debug) are LUMI names --
# see the README for what they mean if your cluster calls them something else.
# -----------------------------------------------------------------------------

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [ -n "$SR_MODULEPATH" ] && command -v module >/dev/null 2>&1; then
    module use "$SR_MODULEPATH"
fi
if [ -n "$SR_MODULES" ] && command -v module >/dev/null 2>&1; then
    module load $SR_MODULES
fi

# scripts live in training/, validation/ and production/ while config.py and the
# other shared modules sit at the repo root, so the root must be importable
export PYTHONPATH=$REPO${SR_VENV:+:$SR_VENV/lib/$SR_PY/site-packages}${PYTHONPATH:+:$PYTHONPATH}

# Per-job scratch dirs.  The MIOpen ones matter on AMD GPUs only, and are harmless
# elsewhere: concurrent ROCm jobs sharing one MIOpen cache corrupt each other's
# kernel database.  MPLCONFIGDIR keeps matplotlib from writing into $HOME.
SR_TMP=${SR_TMP:-$REPO/tmp}
mkdir -p "$SR_TMP"
export MPLCONFIGDIR=$(mktemp -d "$SR_TMP/mplconfig_XXXXXX")
export MIOPEN_USER_DB_PATH=$(mktemp -d "$SR_TMP/miopen_db_XXXXXX")
export MIOPEN_CUSTOM_CACHE_DIR=$(mktemp -d "$SR_TMP/miopen_cache_XXXXXX")

cleanup() {
    rm -rf "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR" "$MPLCONFIGDIR"
}
trap cleanup EXIT
