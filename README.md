# Super_Resolution_FOCCUS
Super Resolution of sea surface height for the FOCCUS project.

The goal is to increase the resolution of the DUACS data set consisting of full, low resolution fields of SSH (gap-filled optimal interpolation, 1/8 deg,
resolves ~150 km) with a neural network using the SWOT SSH fields consisting of sparse, high resolution fields of SSH (2 km posting, resolves ~15-20 km) over the Nordic Seas including the Lofoten Basin.

Note on the branches: the original version (CGAN) is in the CGAN branch. With more SWOT data being available, the training of a diffusion model has become possible and allowed to fix the discontinuities between the patches produced by the CGAN. Indeed, with a diffusion model we can use a global diffusion process to make sure the final, full reconstruction is consistent. To do this the patches are chosen with an overlap, and at each step of the de-noising process, for each overlap, the predicted noise to be removed is an average of the predicted noise coming from each tile. This way, the overlapping prediction is pushed to being the same for both patches (while also being consistent with the rest of each patch).

## Training and production steps

5 main steps to reach the final dataset:

    pre-processing        cache_ssha_wh.npy      The SWOT data are whitened (the power across spatial frequencies is equalized)
    stage 1 (deterministic prediction)       runs/baseline_whitened   Outputs a (deterministic) field mu. Only the predictable parts of the input (corresponding to the largest scales) are corrected by this step.
    stage 2 (stochastic prediction)       runs/diffusion_whitened  EDM diffusion on the residuals r = SWOT - mu. Realistic features are generated over the unpredictable parts of the input. 
    sampling      Different possible outputs (also called realizations or members) are generated with parameters --sigma-max 13 --steps 32 --noise-mode fixed --center-residual.
    calibration   Inflation of the EDM members above 200 km with parameter lambda 3.3

## Structure of the repo

    build_dataset.py                       the producer of the training dataset
    training/                              build caches and target, train, mu + their jobs
    production/                            Used to produce the super-resolved 1993-2026 dataset
    validation/                            archive, diagnostics + their jobs
    validation/plots/lam<L>/               one directory per inflation lambda
    notebooks/view_day.ipynb               one day: maps, currents, KE, flags, RMSE

The SWOT and DUACS data can be downloaded using build_dataset.py. 
The trained weights (`runs/baseline_whitened/best.pt`, 8.5 MB, and
`runs/diffusion_whitened/best.pt`, 94.5 MB) are on Zenodo;
The fully super-resolved dataset is available on Zenodo.

For a quick look there is a smaller copy on
[Hugging Face](https://huggingface.co/datasets/AntoineBernigaud/Super_Resolution_FOCCUS):
the two checkpoints, `SR_duacs_total.nc` cut to the test period, and the SWOT truth
for the same days.  `python download_data.py` fetches all three into the places the
code expects (`--what weights` / `dataset` / `truth` for one at a time).

## Dependencies and running outside of LUMI

Requires torch and Python 3.11.
Dependencies can be install with
pip install -r requirements.txt

Every job script is a SLURM header plus a single `srun python ...` line, so
without SLURM you can just run that line. To submit a job on a supercomputer, adapt:
- **The SLURM account** (not in the job scripts): `export SBATCH_ACCOUNT=project_XXXXXXXXX`.
- **The partitions**: One GPU with >= 32 GB is comfortable (stage 2 trains at batch 32 on 96x96 patches);
full-field sampling is tiled and fits in much less.
- **`env.sh`** has a SITE block at the top: `SR_MODULEPATH` / `SR_MODULES` for a module
  system, `SR_VENV` / `SR_PY` for a venv or conda prefix, `SR_TMP` for scratch.  Set the
  module and venv variables to empty if python is already on PATH.  The `MIOPEN_*`
  variables matter on AMD only and are harmless elsewhere.

## 0. Building the dataset

First modify the first arguments in build_dataset.py. The values by default where the one used for the production of the final dataset.
- TEST_ONLY can be set to True to only download data during the testing period defined by TEST_START and TEST_END.
- If TEST_ONLY is set to False, it will download the data between the dates DATE_START and DATE_END.
- LON_RANGE and LAT_RANGE to select the area of your choice.
- DOWNLOAD_SWOT and DOWNLOAD_DUACS can be set to 1 to download the corresponding data and 0 otherwise.
- To download from CMEMS (for DUACS) and AVISO (for SWOT) you need to enter you credentials in:
AVISO_USER = ""
AVISO_PASS = ""
CMEMS_USER = ""
CMEMS_PASS = ""

build_dataset.py can then called with python 'build_dataset.py all'.

Remark: the network used to produce the final dataset was trained, validated and tested using the following dates:

| split | start | end |
| -------- | -------- | -------- |
| train     |    2023-07-26      |    2025-02-28      |
| val    |    2025-03-21      |    2025-06-30      |
| test    |   2025-07-21       |    2025-11-17      |


## 1. Try the network on a short period

The quickest useful thing: build a few months of data, sample the model on it, and
look at the result.  No training -- it uses the published weights.

    # in build_dataset.py set TEST_ONLY = True (and TEST_START / TEST_END), then
    python build_dataset.py all              # -> sr_dataset/sr_duacs_to_swot_<period>.nc
    sbatch training/job_index.sh             # patch_index.npz (will use the precomputed statistics of the training period for normalization in norm_stats.npz)
    python training/build_cache.py           # cache_ssha.npy, cache_sla.npy
    sbatch training/job_target.sh            # cache_ssha_wh.npy
    sbatch training/job_mu.sh                # mu_whitened.npy, from the downloaded stage 1
    sbatch validation/job_archive.sh --split test --days 10 --out archive_try
    sbatch validation/job_validate.sh 3.3    # -> validation/plots/lam3.3/
    jupyter lab notebooks/view_day.ipynb     # set DAY, run all

## 2. Retrain the model

Build the full record (`TEST_ONLY = False`, `DATE_START` / `DATE_END` in
`build_dataset.py`)

    python build_dataset.py all
    sbatch training/job_index.sh --recompute-stats # Will recompute normalizations statistics in norm_stats.npz)
    python training/build_cache.py
    sbatch training/job_target.sh            # whitened target (the one that works)
    sbatch training/job_baseline.sh          # stage 1 -> runs/baseline_whitened
    sbatch training/job_mu.sh                # mu over the record
    sbatch training/job_diffusion.sh         # stage 2 -> runs/diffusion_whitened
    for S in 10 13 16 20; do                 # re-fit sigma_max
        python validation/diag_patch_psd.py --sigma-max $S --out psd_smax$S
    done
    sbatch validation/job_archive.sh         # -> archive_wh13
    sbatch validation/job_validate.sh 3.3

## 3. Validate

`job_validate.sh <lambda>` runs the whole suite on an archive and writes
`validation/plots/lam<lambda>/`: CRPS and rank histograms, RMSE, coherence with SWOT,
swath-geometry spectra and cross-scale transfer, the phase-surrogate test, the offset
diagnostics and daily maps.  Lambda being the inflation parameter.

Spectra and cross-scale transfer pooled over the whole record:

    sbatch validation/job_total.sh 3.3       # train + val + test
    sbatch validation/job_fullperiod.sh      # 1993-2026, one array task per year
    sbatch validation/job_fullperiod_combine.sh

If you downloaded `SR_duacs_total.nc` instead of running the model, everything above
still works.  Turn its days into the archive format first:

    python validation/archive_from_dataset.py --dataset SR_duacs_total.nc \
           --split test --days 40 --out archive_wh13
    sbatch validation/job_validate.sh 1.0    # its members are ALREADY inflated at 3.3
    python validation/swath_fullperiod.py --dataset SR_duacs_total.nc --year 2020

Scoring against SWOT needs the truth alongside it -- `python download_data.py --what
truth` for the test period, or `build_dataset.py` for the whole record.  The notebook
and the spectra work without it, and simply skip the SWOT panels.

## Details

- `sigma_max` is an optimized hyper-parameter, fitted with
`validation/diag_patch_psd.py --sigma-max`. Tt is fitted by matching that network's 10-60 km power to the
target's.

- There are three possible value for the inflation parameter lambda:
      - lambda = 3.3 flattens the rank-histogram (option retained for the production of the fully super-resolved dataset).
      - lambda = 4.6 minimizes the spread-skill.
      - lambda = 5 minimizes the CRPS.

- During inference, each member uses a fix random noise across days to maintain coherence over time.


