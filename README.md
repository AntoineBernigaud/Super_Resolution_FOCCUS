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
    regions.py  device.py                  scoring regions; CPU/GPU selection
    build_dataset.py + KaRIn_*.geojson     the producer of the training dataset
    training/                              build caches and target, train, mu + their jobs
    production/                            the super-resolved 1993-2026 dataset
    validation/                            archive, diagnostics + their jobs
    validation/plots/lam<L>/               one directory per inflation lambda
    notebooks/view_day.ipynb               one day: maps, currents, KE, flags, RMSE

The data itself is not in the repository (see below); the paths above are where the
jobs expect to find or write it.

## Running

    python build_dataset.py all          # -> sr_dataset/sr_duacs_to_swot_<period>.nc
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

`build_dataset.py` must run first of all (it downloads SWOT from AVISO and DUACS from
Copernicus Marine and grids them); then `job_index`; and `job_mu` between the two
training stages.  Without SLURM, each job script is a header plus one
`srun python ...` line -- run that line directly.

Two checks, on the produced data rather than on the code:

    sbatch production/job_check_record.sh    # every day present once, openable, right
                                             # shapes, no gaps, sane per-year statistics
    sbatch production/job_merge.sh --verify  # the merged file against the per-day
                                             # checksums -- run BEFORE deleting product/

## Details

- `sigma_max` is an optimized hyper-parameter, fitted with
`validation/diag_patch_psd.py --sigma-max`.  It belongs to the TRAINED NETWORK, not
to the target: it is fitted by matching that network's 10-60 km power to the
target's, and three trainings of this architecture needed 13, 16 and 11.  Re-fit it
after any retraining, before building an archive, or the comparison is confounded.

- There are three possible value for the inflation parameter lambda:
      - lambda = 3.3 flattens the the rank-histogram (option retained for the production of the fully super-resolved dataset).
      - lambda = 4.6 minimizes the spread-skill.
      - lambda = 5 minimizes the CRPS.
  The three criteria do not agree on one width, which is itself the finding: a
  correctly shaped ensemble would calibrate all three at the same lambda.

- Each member uses a fix random noise across days to maintain coherence over time.

## The super-resolved dataset (`production/`)

    sbatch production/job_produce.sh     # full record, ~442 GPU-hours

Input `DUACS_full.nc`: DUACS L4 `sla`, 1993-01-01 .. 2026, 12,069 days.
Output `product/YYYY/sr_nordic_sla_YYYYMMDD.nc` (~9.7 MB/day), merged by
`sbatch production/job_merge.sh` into the single `SR_duacs_total.nc` (~108 GB)
published on Zenodo.

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
| `sr_dataset/sr_duacs_to_swot_<period>.nc` | 0.8 GB | DUACS/SWOT training pairs: `python build_dataset.py all` |
| `downloads*/Science/` | ~1.2 GB | SWOT L3 LR SSH Expert v2.0.1 passes, downloaded by the same command; kept, because the swath geometry is read from them |
| `cache_*.npy`, `mu_whitened.npy`, `patch_index.npz` | ~4.5 GB | regenerated: `training/build_cache.py`, `job_index.sh`, `job_target.sh`, `job_mu.sh` |
| `archive_*/`, `validation/plots/*/daily/` | ~6 GB | regenerated: `validation/job_archive.sh`, `job_validate.sh` |

`norm_stats.npz` IS included (2.4 KB): it holds the normalisation constants the
network was trained with, so inference works without the training dataset.  So are
the small results needed to rebuild the target, and the summary figures:
`runs/native_vs_collocated/native_vs_collocated.json` (the measured gridding artifact
the whitened target is built from), the training histories, and every validation
figure except the daily maps.

## Running outside LUMI

Python 3.11, `pip install -r requirements.txt`.  Install `torch` first, matching your
accelerator (ROCm / CUDA / CPU wheels -- see the top of that file).  `device.py`
selects the GPU when there is one and falls back to the CPU with autocast off, so
nothing is hard-wired to AMD.

Three site-specific things, all outside the python:

- **The SLURM account** is not in the job scripts: `export SBATCH_ACCOUNT=project_XXXXXXXXX`.
- **The partitions** `small-g` (1 GPU), `small` (CPU) and `debug` (CPU, short) are
  LUMI names -- edit the `#SBATCH --partition=` lines.
- **`env.sh`** has a SITE block at the top: `SR_MODULEPATH` / `SR_MODULES` for a module
  system, `SR_VENV` / `SR_PY` for a venv or conda prefix, `SR_TMP` for scratch.  Set the
  module and venv variables to empty if python is already on PATH.  The `MIOPEN_*`
  variables matter on AMD only and are harmless elsewhere.

One GPU with >= 32 GB is comfortable (stage 2 trains at batch 32 on 96x96 patches);
full-field sampling is tiled and fits in much less.

## Using the published dataset without running the model

Downloading `SR_duacs_total.nc` from Zenodo is enough to look at the product and to
recompute its spectra and cross-scale transfer -- no GPU, no weights, no training data:

    jupyter lab notebooks/view_day.ipynb                       # set DAY, run all
    python validation/swath_fullperiod.py --dataset SR_duacs_total.nc --year 2020
    python validation/swath_fullperiod.py --combine

Scoring it against SWOT additionally needs the training dataset, since SWOT is the
truth.  With that in place, turn dataset days into the archive format every
diagnostic reads:

    python validation/archive_from_dataset.py --dataset SR_duacs_total.nc \
           --split test --days 40 --out archive_wh13
    sbatch validation/job_validate.sh 1.0    # its members are ALREADY inflated at 3.3
