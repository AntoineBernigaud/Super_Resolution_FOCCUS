# FOCCUS_ssh_SR

Diffusion super-resolution from **DUACS** (gap-filled optimal interpolation, 1/8 deg,
resolves ~150 km) to **SWOT** (2 km posting, resolves ~15-20 km) over the Nordic Seas.

Version 2 of this repository.  Version 1 (a conditional GAN, release `v1`) is kept
on the [`CGAN`](../../tree/CGAN) branch.

This repository is the **single best configuration only**, extracted from the
exploratory tree `FOCCUS_C_V0`.  Closed branches (stage-0 coarse diffusion, native-L3
supervision, binned and low-pass targets, S_churn, the regularisation and ambient
sweeps) are deliberately **not** here; `FOCCUS_C_V0/CLAUDE.md` holds the evidence for
why each was abandoned.

## The configuration

    target        cache_ssha_wh.npy      whitened, not low-passed
    stage 1       runs/baseline_whitened   deterministic mu
    stage 2       runs/diffusion_whitened  EDM diffusion on r = y - mu
    sampling      --sigma-max 13 --steps 32 --noise-mode fixed --center-residual
    calibration   post-hoc, scale-selective: --above-km 200, lambda 5.0

## Layout

    config.py data.py nets.py edm.py ...   shared modules; config.ROOT is THIS directory
    inflation.py                           scale-selective inflation (validation + production)
    training/                              build, train, smoke-test + their jobs
    production/                            the super-resolved 1993-2026 dataset
    validation/                            archive, diagnostics + their jobs
    validation/plots/lam<L>/               one directory per inflation lambda
    *.npy *.nc swath_geometry/ runs/       SYMLINKS into FOCCUS_C_V0 (nothing copied)

Data is symlinked, not duplicated: the caches alone are 1.5 GB each and the swath
geometry is 1.2 GB.

## Running

Never on the login node -- everything goes through `sbatch`, including one-off
inspection.  Submit from the repo root (or any subdirectory: the job scripts walk up
from `$SLURM_SUBMIT_DIR` to the root that holds `env.sh`).

    sbatch training/job_chkpaths.sh      # paths + imports + smoke test.  Run first.
    sbatch training/job_index.sh         # patch_index.npz + norm_stats.npz
    sbatch training/job_target.sh        # cache_ssha_wh.npy
    sbatch training/job_baseline.sh      # stage 1  -> runs/baseline_whitened
    sbatch training/job_mu.sh            # mu_whitened.npy   (BETWEEN the stages)
    sbatch training/job_diffusion.sh     # stage 2  -> runs/diffusion_whitened
    sbatch validation/job_archive.sh     # -> archive_wh13   (sampling, ~3 h)
    sbatch validation/job_validate.sh 5.0    # -> validation/plots/lam5.0
    sbatch validation/job_validate.sh 4.6
    sbatch validation/job_validate.sh 3.3
    sbatch validation/job_validate.sh 1.0    # uninflated

`job_index` must precede everything; `job_mu` must run between the two stages.

## Five things that will bite you

**`sigma_max` belongs to the trained network, not to the target.**  13 is fitted to
`runs/diffusion_whitened` by matching its 10-60 km MultiDiffusion/target power to 1.0.
Three trainings of the same architecture gave 13, 16 and 11.  Refit it with
`validation/diag_patch_psd.py --sigma-max` after ANY retrain; inheriting it silently
confounds every comparison that follows.

**Inflation is calibration, not skill, and it is scale-SELECTIVE.**  lambda 5.0 applies
only above 200 km.  Scale-blind inflation reaches the same CRPS and multiplies the
27-74 km bands by 4-5x.  Read CRPS, spread-skill and rank histograms from an inflated
archive; read spectra, EKE, transfer and bicoherence from `archive_wh13` at lambda 1.
The inflated members overshoot 111-223 km by 1.7-2.5x by construction.

**The three lambdas do not agree, and that is the point.**  CRPS is minimised at 5.0,
spread-skill reaches 1 at ~4.6, and the rank-histogram ends ratio reaches 1 at ~3.3.
A correctly shaped ensemble would calibrate all three at one width.  This one does not,
because every member shares a single frozen mu and the added spread therefore carries
no information.  `validation/plots/lam{5.0,4.6,3.3}` exist so that disagreement is
visible rather than asserted.

**Do not rank anything by FSS.**  `fss_analysis.py` is present because it defines
`REGIONS`/`region_slices`, which most diagnostics import.  The worst model ever trained
in this project has the best FSS of any run.  Report on the `global` region only.

**Use a single member for anything involving derivatives** -- currents, strain,
cascade.  The ensemble mean is not a realisation and its gradient statistics are wrong
by construction.  `plot_day_geo` plots the mean of the member SPEEDS, which is 1.61x
the speed of the mean field; the panel says so.

## The super-resolved dataset (`production/`)

    sbatch production/job_test.sh        # 4 test days + checks against the validated archive
    sbatch production/job_produce.sh     # full record, 32-shard array, ~442 GPU-hours

Input `DUACS_full.nc`: DUACS L4 `sla`, 1993-01-01 .. 2026, 12,069 days, bit-identical to
the training input on all 828 overlapping days.  Output `product/YYYY/sr_nordic_sla_YYYYMMDD.nc`,
one file per day (~9.7 MB), CF-1.8, laid out like CMEMS DUACS L4 daily files and encoded
like SWOT L3 (`int32`, `scale_factor` 1e-4 m, zlib + shuffle):

| variable | dims | content |
|---|---|---|
| `sla_duacs` | time, duacs_latitude, duacs_longitude | DUACS input, native 1/8 deg, unmodified |
| `sla_mu` | time, latitude, longitude | stage-1 deterministic mean -- the best single map |
| `sla` | time, realization, latitude, longitude | 8 diffusion members, inflated lambda 3.3 above 200 km |
| `sla_mean` | time, latitude, longitude | mean of the 8 members -- NOT a realisation |
| `quality_flag` | time, latitude, longitude | CF bitmask, below |

`quality_flag` carries only conditions with MEASURED degradation or provenance:

| bit | meaning | evidence |
|---|---|---|
| 1 | `near_coast` (< 25 km of DUACS land) | CRPS skill over DUACS 28% offshore, 16% at 10-25 km, 5% within 10 km |
| 2 | `input_out_of_distribution` | day's DUACS gradient energy below the training-period 1st percentile; input is ~0.5x training in 1993-1999, in range again from 2017 |
| 4 | `no_swot_validation` | before 2023-07-26: nothing independent to validate against |
| 8 | `in_training_period` | 2023-07-26 .. 2025-02-28: agreement with SWOT is not an independent test |

Latitude was measured and NOT flagged (25% skill north of 74N against 29-31% in the
open Nordic Seas -- real, but not a quality break).  There is no sea-ice bit: the
DUACS file carries no ice information and its ocean mask never changes.

## What is not in this repository

Data, model weights and large outputs are excluded by `.gitignore`:

| item | size | where it comes from |
|---|---|---|
| `SR_duacs_total.nc` | ~108 GB | the super-resolved dataset -- Zenodo |
| `runs/*/best.pt` | 8.5 MB + 94.5 MB | trained stage-1 / stage-2 weights -- Zenodo |
| `DUACS_full.nc` | 1.2 GB | CMEMS `cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D`, `sla`, 62-78N, 18W-20E |
| `sr_duacs_to_swot.nc` | 0.8 GB | DUACS/SWOT training pairs, built by the producer pipeline (see its dataset README) |
| `swath_geometry/` | 1.2 GB | SWOT L3 LR SSH Expert v2.0.1 pass files (AVISO) + `swath_geometry_index.csv` |
| `cache_*.npy`, `mu_whitened.npy`, `patch_index.npz`, `norm_stats.npz` | ~4.5 GB | regenerated: `training/job_index.sh`, `job_target.sh`, `job_mu.sh` |
| `archive_*/`, `validation/plots/*/daily/` | ~6 GB | regenerated: `validation/job_archive.sh`, `job_validate.sh` |

The small results needed to rebuild the target and the summary figures ARE included:
`runs/native_vs_collocated/native_vs_collocated.json` (the measured gridding artifact
the whitened target is built from), the training histories, and every validation
figure except the daily maps.

## Running outside LUMI

The job scripts are SLURM scripts for LUMI.  Two things are site-specific:
`env.sh` (the CSC `pytorch` module and a venv providing `xarray`/`netCDF4`), and the
`#SBATCH --account=` / `--partition=` lines (LUMI project and `small-g` / `small` /
`debug` partitions).  Adapt those two; nothing else assumes LUMI.  Python
dependencies: `torch`, `numpy`, `scipy`, `netCDF4`, `xarray`, `matplotlib`.
