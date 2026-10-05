"""Upload the SWOT truth subset to the Hugging Face dataset repository.

    python production/upload_truth_hf.py

Producer-side only -- a user of the published data never runs this; they run
`download_data.py` at the repository root, which expects the file at exactly the
remote path this script writes to.

It uploads one file:

    sr_dataset/sr_duacs_to_swot_test_period.nc   (106 MB, 119 days, made by
                                                  production/extract_period.py)
          -> data/sr_duacs_to_swot_test_period.nc

Needs `huggingface_hub` and a WRITE token.  Either run `hf auth login` once, or
export HF_TOKEN.  The token is never printed and never stored by this script.

Hugging Face stores anything over 10 MB through git-lfs automatically, so no
.gitattributes work is needed on the dataset repo.
"""
import argparse
import sys
from pathlib import Path

REPO = "AntoineBernigaud/Super_Resolution_FOCCUS"
LOCAL = "sr_dataset/sr_duacs_to_swot_test_period.nc"
REMOTE = "data/sr_duacs_to_swot_test_period.nc"
ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=LOCAL, help=f"local file (default {LOCAL})")
    ap.add_argument("--remote", default=REMOTE,
                    help=f"path inside the dataset repo (default {REMOTE})")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--yes", action="store_true", help="skip the confirmation")
    args = ap.parse_args()

    src = Path(args.file)
    if not src.is_absolute():
        src = ROOT / src
    if not src.exists():
        sys.exit(f"not found: {src}\n"
                 "Build it first:\n"
                 "  python production/extract_period.py --src sr_duacs_to_swot.nc "
                 "--split test --out sr_dataset/sr_duacs_to_swot_test_period.nc")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("huggingface_hub is not installed:\n"
                 "  pip install --user huggingface_hub\n"
                 "  export PATH=$HOME/.local/bin:$PATH")

    mb = src.stat().st_size / 1e6
    print(f"{src}  ({mb:.1f} MB)")
    print(f"  -> {args.repo} : {args.remote}  (repo_type=dataset)")
    if not args.yes:
        if input("upload? [y/N] ").strip().lower() not in ("y", "yes"):
            sys.exit("aborted")

    api = HfApi()
    try:
        who = api.whoami()["name"]
    except Exception as e:                      # no token, or an expired one
        sys.exit(f"not authenticated ({e}).\nRun `hf auth login` with a WRITE token, "
                 "or export HF_TOKEN.")
    print(f"authenticated as {who}, uploading...")

    url = api.upload_file(path_or_fileobj=str(src), path_in_repo=args.remote,
                          repo_id=args.repo, repo_type="dataset",
                          commit_message=f"Add {Path(args.remote).name}: SWOT truth "
                                         "for the published test period")
    print(f"done: {url}")
    print("\nCheck it from a clean tree with:\n"
          "  python download_data.py --what truth")


if __name__ == "__main__":
    sys.exit(main())
