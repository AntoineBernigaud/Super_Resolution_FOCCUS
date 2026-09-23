# Super_Resolution_FOCCUS
Super Resolution of sea surface height for the FOCCUS project.

The goal is to increase the resolution of the DUACS data set consisting of full, low resolution fields of SSH (gap-filled optimal interpolation, 1/8 deg,
resolves ~150 km) with a neural network using the SWOT SSH fields consisting of sparse, high resolution fields of SSH (2 km posting, resolves ~15-20 km) over the Nordic Seas including the Lofoten Basin.

Note on the branches: the original version (CGAN) is in the CGAN branch. With more SWOT data being available, the training of a diffusion model has become possible and allowed to fix the discontinuities between the patches produced by the CGAN. Indeed, with a diffusion model we can use a global diffusion process to make sure the final, full reconstruction is consistent. To do this the patches are chosen with an overlap, and at each step of the de-noising process, for each overlap, the predicted noise to be removed is an average of the predicted noise coming from each tile. This way, the overlapping prediction is pushed to being the same for both patches (while also being consistent with the rest of each patch).

## The configuration

    target        cache_ssha_wh.npy      The SWOT data are whitened (the power across spatial frequencies is equalized)
    stage 1       runs/baseline_whitened   Outputs a deterministic field mu
    stage 2       runs/diffusion_whitened  EDM diffusion on the residuals r = y - mu
    sampling      --sigma-max 13 --steps 32 --noise-mode fixed --center-residual
    calibration   inflation of the EDM members post-hoc, scale-selective: --above-km 200, lambda 3.3

## Layout

    config.py data.py nets.py edm.py ...   shared modules
    inflation.py                           scale-selective inflation (validation + production)
    training/                              build, train, smoke-test + their jobs
    production/                            the super-resolved 1993-2026 dataset
    validation/                            archive, diagnostics + their jobs
    validation/plots/lam<L>/               one directory per inflation lambda
    *.npy *.nc swath_geometry/ runs/       SYMLINKS into FOCCUS_C_V0 (nothing copied)

Data is symlinked, not duplicated: the caches alone are 1.5 GB each and the swath
geometry is 1.2 GB.

## Running

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

## Details

- `sigma_max` is an optimized hyper-parameter, fitted with
`validation/diag_patch_psd.py --sigma-max`.

- There are three possible value for the inflation parameter lambda:
      - lambda = 3.3 flattens the the rank-histogram (option retained for the production of the fully super-resolved dataset).
      - lambda = 4.6 minimizes the spread-skill.
      - lambda = 5 minimizes the CRPS do not agree, and that is the point.**  CRPS is minimised at 5.0.

- Each member uses a fix random noise across days to maintain coherence over time.

## The super-resolved dataset (`production/`)

    sbatch production/job_produce.sh     # full record, ~442 GPU-hours

Input `DUACS_full.nc`: DUACS L4 `sla`, 1993-01-01 .. 2026, 12,069 days
Output `product/YYYY/sr_nordic_sla_YYYYMMDD.nc`,

| variable | dims | content |
|---|---|---|
| `sla_duacs` | time, duacs_latitude, duacs_longitude | DUACS input, native 1/8 deg (unmodified) |
| `sla_mu` | time, latitude, longitude | stage-1 deterministic mean |
| `sla` | time, realization, latitude, longitude | 8 diffusion members, inflated lambda 3.3 above 200 km |
| `sla_mean` | time, latitude, longitude | mean of the 8 members |
| `quality_flag` | time, latitude, longitude | CF bitmask, below |

`quality_flag` carries only conditions with MEASURED degradation or provenance:

| bit | meaning | reason |
|---|---|---|
| 1 | `near_coast` (< 25 km of DUACS land) | The CRPS skill is better offshore: 28% better over DUACS offshore, 16% at 10-25 km, 5% within 10 km |
| 2 | `input_out_of_distribution` | DUACS gradient energy below the training-period 1st percentile for this day |
| 4 | `no_swot_validation` | before 2023-07-26: nothing independent to validate against |
| 8 | `in_training_period` | 2023-07-26 .. 2025-02-28: agreement with SWOT is not an independent test |


## What is not in this repository

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
