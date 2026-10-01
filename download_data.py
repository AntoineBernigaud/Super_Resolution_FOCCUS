"""Download the trained weights and the sample dataset from Hugging Face.

    https://huggingface.co/datasets/AntoineBernigaud/Super_Resolution_FOCCUS

    python download_data.py              # weights + dataset + SWOT truth
    python download_data.py --what weights
    python download_data.py --what dataset
    python download_data.py --what truth

It puts every file where the code already expects it:

    baseline_whitened/best.pt   -> runs/baseline_whitened/best.pt    (stage 1)
    diffusion_whitened/best.pt  -> runs/diffusion_whitened/best.pt   (stage 2)
    data/SR_duacs_total.nc      -> SR_duacs_total.nc                 (the test period)
    data/sr_duacs_to_swot_test_period.nc
                                -> sr_dataset/sr_duacs_to_swot_test_period.nc

Together with `norm_stats.npz`, which ships with the repository, the two checkpoints
are everything inference needs -- no caches, no training dataset.  The stage-2 file
also carries `r_scale`, the amplitude of the sampled residual, so it must be kept
whole.

The third file is the SWOT truth for the same 119 days (gridded SWOT L3 `ssha` and
the DUACS `sla` that goes with it).  It is NOT needed to run the model or to look at
the product; it is what the notebook and the validation scripts compare against, and
without it every SWOT panel is skipped.

Uses `huggingface_hub` when it is installed (resumable, cached, checksum-checked) and
falls back to a plain streamed HTTPS download otherwise.
"""
import argparse
import sys
import urllib.request
from pathlib import Path

REPO = "AntoineBernigaud/Super_Resolution_FOCCUS"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
FILES = {
    "weights": [("baseline_whitened/best.pt", "runs/baseline_whitened/best.pt"),
                ("diffusion_whitened/best.pt", "runs/diffusion_whitened/best.pt")],
    "dataset": [("data/SR_duacs_total.nc", "SR_duacs_total.nc")],
    "truth": [("data/sr_duacs_to_swot_test_period.nc",
               "sr_dataset/sr_duacs_to_swot_test_period.nc")],
}
ORDER = ["weights", "dataset", "truth"]


def human(n):
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def stream(remote, dst):
    """Plain HTTPS download, straight to a .part file so a broken one is obvious."""
    part = dst.with_suffix(dst.suffix + ".part")
    # \r only when a terminal is watching; piped or in a log it would be one long line
    tty = sys.stdout.isatty()
    with urllib.request.urlopen(f"{BASE}/{remote}") as r, open(part, "wb") as fh:
        total = int(r.headers.get("Content-Length", 0))
        done = step = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if not total:
                continue
            pct = 100 * done / total
            if tty:
                print(f"\r    {human(done)} / {human(total)} ({pct:5.1f}%)",
                      end="", flush=True)
            elif pct >= step:
                print(f"    {human(done)} / {human(total)} ({pct:5.1f}%)", flush=True)
                step += 25
    if tty:
        print()
    part.rename(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="all",
                    choices=["all", "weights", "dataset", "truth"])
    ap.add_argument("--dest", default=Path(__file__).resolve().parent,
                    help="repository root (default: where this script lives)")
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args()

    dest = Path(args.dest)
    want = ([f for k in ORDER for f in FILES[k]] if args.what == "all"
            else FILES[args.what])
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        hf_hub_download = None
        print("huggingface_hub not installed -- falling back to plain HTTPS "
              "(pip install huggingface_hub for resumable, checksum-checked "
              "downloads)\n")

    for remote, local in want:
        out = dest / local
        if out.exists() and not args.force:
            print(f"  {local} already there ({human(out.stat().st_size)}), skipping")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"  {remote} -> {local}")
        if hf_hub_download is not None:
            p = hf_hub_download(repo_id=REPO, filename=remote, repo_type="dataset")
            # copy out of the hub cache so the tree is self-contained
            import shutil
            shutil.copyfile(p, out)
        else:
            stream(remote, out)
        print(f"    {human(out.stat().st_size)}")

    print("\nReady.  Inference needs only the two checkpoints and norm_stats.npz:")
    print("  python production/produce_sr.py --start 2025-07-21 --end 2025-07-31 "
          "--out product")
    print("Or look at the downloaded dataset without running the model:")
    print("  jupyter lab notebooks/view_day.ipynb")
    print("The SWOT truth covers 2025-07-21 .. 2025-11-17 (119 days: the record has "
          "no 2025-10-28);\nany day outside it still plots, without the SWOT panels.")


if __name__ == "__main__":
    sys.exit(main())
