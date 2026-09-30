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

No data is in the repository; the paths above are where the jobs expect to find or
write it.  `norm_stats.npz` is the exception and is tracked: two constants, part of
the trained model (see step 1).

## Installing

Python 3.11, then `pip install -r requirements.txt`.  Install `torch` first, matching
your accelerator (ROCm / CUDA / CPU wheels -- see the top of that file).  `device.py`
uses the GPU when there is one and falls back to the CPU with autocast off.

Credentials: `build_dataset.py` downloads SWOT from the AVISO FTP (`AVISO_USER`,
`AVISO_PASS`) and DUACS from Copernicus Marine (`copernicusmarine login`, or
`COPERNICUSMARINE_SERVICE_USERNAME` / `_PASSWORD`).

The trained weights (`runs/baseline_whitened/best.pt`, 8.5 MB, and
`runs/diffusion_whitened/best.pt`, 94.5 MB) and the finished product
(`SR_duacs_total.nc`, ~108 GB) are on Zenodo; no data is in this repository.

Every job script below is a SLURM header plus a single `srun python ...` line, so
without SLURM just run that line.

## 1. Try the network on a short period

The quickest useful thing: build a few months of data, sample the model on it, and
look at the result.  No training -- it uses the published weights.

    # in build_dataset.py set TEST_ONLY = True (and TEST_START / TEST_END), then
    python build_dataset.py all              # -> sr_dataset/sr_duacs_to_swot_<period>.nc
    sbatch training/job_index.sh             # patch_index.npz (keeps norm_stats.npz)
    python training/build_cache.py           # cache_ssha.npy, cache_sla.npy
    sbatch training/job_target.sh            # cache_ssha_wh.npy
    sbatch training/job_mu.sh                # mu_whitened.npy, from the downloaded stage 1
    sbatch validation/job_archive.sh --split test --days 10 --out archive_try
    sbatch validation/job_validate.sh 3.3    # -> validation/plots/lam3.3/
    jupyter lab notebooks/view_day.ipynb     # set DAY, run all

Keep `norm_stats.npz` as it ships: it holds the constants the published weights were
trained with, and `job_index` no longer overwrites it.  A short period has its own
mean and standard deviation, and normalising with those would feed the network
something it was never trained on.

## 2. Retrain the model

Build the full record (`TEST_ONLY = False`, `DATE_START` / `DATE_END` in
`build_dataset.py`), then compute your own normalisation, because now the constants
must match the data you train on:

    python build_dataset.py all
    sbatch training/job_index.sh --recompute-stats
    python training/build_cache.py
    sbatch training/job_target.sh            # whitened target (the one that works)
    sbatch training/job_baseline.sh          # stage 1 -> runs/baseline_whitened
    sbatch training/job_mu.sh                # mu over the record   (BETWEEN the stages)
    sbatch training/job_diffusion.sh         # stage 2 -> runs/diffusion_whitened
    for S in 10 13 16 20; do                 # re-fit sigma_max -- not optional
        python validation/diag_patch_psd.py --sigma-max $S --out psd_smax$S
    done
    sbatch validation/job_archive.sh         # -> archive_wh13
    sbatch validation/job_validate.sh 3.3

`job_index` first, `job_mu` between the two training stages.  Re-fitting `sigma_max`
is not optional -- see Details.

## 3. Validate

`job_validate.sh <lambda>` runs the whole suite on an archive and writes
`validation/plots/lam<lambda>/`: CRPS and rank histograms, RMSE, coherence with SWOT,
swath-geometry spectra and cross-scale transfer, bicoherence, the offset diagnostics
and daily maps.  It inflates the archive itself, so give it the lambda you want.

Spectra and cross-scale transfer pooled over the whole record:

    sbatch validation/job_total.sh 3.3       # train + val + test
    sbatch validation/job_fullperiod.sh      # 1993-2026, one array task per year
    sbatch validation/job_fullperiod_combine.sh

If you downloaded `SR_duacs_total.nc` instead of running the model, everything above
still works -- no GPU, no weights.  Turn its days into the archive format first:

    python validation/archive_from_dataset.py --dataset SR_duacs_total.nc \
           --split test --days 40 --out archive_wh13
    sbatch validation/job_validate.sh 1.0    # its members are ALREADY inflated at 3.3
    python validation/swath_fullperiod.py --dataset SR_duacs_total.nc --year 2020

Scoring against SWOT needs the training dataset too, since SWOT is the truth; the
notebook and the spectra work without it.

## 4. Produce the full dataset

`production/` holds the scripts that made the published 1993-2026 product from
`DUACS_full.nc` (`job_produce.sh`, then `job_merge.sh` to merge the per-day files
into `SR_duacs_total.nc`, plus two integrity checks).  It is ~442 GPU-hours and
~108 GB; the result is on Zenodo, so you only need this to rebuild it.

## Details

- `sigma_max` is an optimized hyper-parameter, fitted with
`validation/diag_patch_psd.py --sigma-max`.  It belongs to the TRAINED NETWORK, not
to the target: it is fitted by matching that network's 10-60 km power to the
target's, and three trainings of this architecture needed 13, 16 and 11.  Re-fit it
after any retraining, before building an archive, or the comparison is confounded.

- There are three possible value for the inflation parameter lambda:
      - lambda = 3.3 flattens the rank-histogram (option retained for the production of the fully super-resolved dataset).
      - lambda = 4.6 minimizes the spread-skill.
      - lambda = 5 minimizes the CRPS.
  The three criteria do not agree on one width, which is itself the finding: a
  correctly shaped ensemble would calibrate all three at the same lambda.

- Each member uses a fix random noise across days to maintain coherence over time.

## Running outside LUMI

Nothing in the python is hard-wired to AMD or to LUMI.  Three site-specific things,
all outside it:

- **The SLURM account** is not in the job scripts: `export SBATCH_ACCOUNT=project_XXXXXXXXX`.
- **The partitions** `small-g` (1 GPU), `small` (CPU) and `debug` (CPU, short) are
  LUMI names -- edit the `#SBATCH --partition=` lines.
- **`env.sh`** has a SITE block at the top: `SR_MODULEPATH` / `SR_MODULES` for a module
  system, `SR_VENV` / `SR_PY` for a venv or conda prefix, `SR_TMP` for scratch.  Set the
  module and venv variables to empty if python is already on PATH.  The `MIOPEN_*`
  variables matter on AMD only and are harmless elsewhere.

One GPU with >= 32 GB is comfortable (stage 2 trains at batch 32 on 96x96 patches);
full-field sampling is tiled and fits in much less.
