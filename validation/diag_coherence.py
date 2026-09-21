"""How much of DUACS's information does mu actually extract -- and is there headroom?

The alignment question in one measurement.  A member's features below ~70 km cannot
match the truth because DUACS does not contain them (coherence^2 0.83 at 111 km, 0.54
at 74 km, 0.01 at 56 km).  But at 70-150 km there IS information, and if mu falls
short of DUACS's own coherence there, stage 1 is leaving skill on the table -- and
every member is built on mu, so fixing it would improve alignment for all of them.

  coh2(DUACS, truth)   the ceiling: everything linearly extractable from the input
  coh2(mu, truth)      what stage 1 actually extracts
  coh2(member, truth)  what one realisation achieves
  coh2(mean of 8)      what averaging achieves

coh2(mu) ~ coh2(DUACS) at every scale => no headroom, alignment is input-limited.
coh2(mu) < coh2(DUACS) in the 70-150 km band => stage 1 can be improved.

A member MUST score below mu: it adds texture uncorrelated with the truth, which is
what a sample is meant to do.  That is not a defect and is not evidence of headroom.

Same machinery as diag_sst.py: Hanning-windowed along-latitude segments inside the
eval box, single-predictor magnitude-squared coherence with the adjusted-R^2 bias
correction, accumulated over the archived days.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import config as C
from data import load_stats

SEG, STEP, RIDGE = 128, 32, 1e-6
DY_KM = C.DLAT_C / C.REFINE_LAT * 111.320


def valid_starts(valid, seg, step):
    """Segment origins whose whole column-run is observed."""
    ok = np.ones_like(valid)
    for k in range(seg):
        ok[:valid.shape[0] - seg + 1] &= valid[k:valid.shape[0] - seg + 1 + k]
    ok = ok[:valid.shape[0] - seg + 1]
    ii, jj = np.nonzero(ok[::step])
    return ii * step, jj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="archive_wh13")
    ap.add_argument("--out", default="validation/plots/lam1.0/coherence")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mean, std = load_stats()
    ssha = np.load(C.CACHE_SSHA, mmap_mode="r")
    sla = np.load(C.CACHE_SLA, mmap_mode="r")
    si, sj = slice(128, 768), slice(312, 912)

    names = ["DUACS", "mu (deterministic)", "member", "mean of 8"]
    nf = SEG // 2 + 1
    Sxx = {k: np.zeros(nf) for k in names}
    Sxy = {k: np.zeros(nf, complex) for k in names}
    Syy = np.zeros(nf)
    nseg = 0
    win = np.hanning(SEG)

    files = sorted(Path(args.archive).glob("pred_*.npz"))
    print(f"{len(files)} archived days from {args.archive}")
    for c, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        t = int(d["t"])
        y = np.asarray(ssha[t], np.float32)[si, sj]
        v = np.isfinite(y)
        ii, jj = valid_starts(v, SEG, STEP)
        if ii.size == 0:
            continue
        rows = ii[:, None] + np.arange(SEG)[None, :]

        def seg_fft(a):
            q = a[rows, jj[:, None]] * win
            return np.fft.rfft(q - q.mean(axis=1, keepdims=True), axis=1)

        yn = (np.clip(np.nan_to_num(y, nan=0.0), -C.CLIP_M, C.CLIP_M) - mean) / std
        duacs = np.repeat(np.repeat(np.asarray(sla[t], np.float32),
                                    C.REFINE_LAT, 0), C.REFINE_LON, 1)[si, sj]
        ens = d["ens"].astype(np.float32)[:, si, sj]
        fields = {"DUACS": (np.nan_to_num(duacs, nan=0.0) - mean) / std,
                  "mu (deterministic)": d["mu"].astype(np.float32)[si, sj],
                  "member": ens[0], "mean of 8": ens.mean(0)}

        FY = seg_fft(yn)
        Syy += (FY * FY.conj()).real.sum(axis=0)
        for k, a in fields.items():
            FA = seg_fft(a)
            Sxx[k] += (FA * FA.conj()).real.sum(axis=0)
            Sxy[k] += (FA * FY.conj()).sum(axis=0)
        nseg += FY.shape[0]
        if c % 10 == 0:
            print(f"  {c + 1}/{len(files)}  segments {nseg}", flush=True)

    freq = np.fft.rfftfreq(SEG, d=DY_KM)
    wl = np.where(freq > 0, 1.0 / np.maximum(freq, 1e-12), np.inf)

    curves = {}
    for k in names:
        g = np.abs(Sxy[k]) ** 2 / np.maximum(Sxx[k] * Syy, 1e-30)
        g = np.clip(g, 0, 1)
        curves[k] = np.clip(1 - (1 - g) * (nseg - 1) / max(nseg - 1, 1), 0, 1)

    print(f"\n{nseg} segments")
    print("\n=== coherence^2 with SWOT, by wavelength ===")
    print(f"{'lambda km':>10}" + "".join(f"{k:>22}" for k in names)
          + f"{'mu / DUACS':>12}")
    for T in (200, 150, 111, 90, 74, 60, 50, 40, 30, 20):
        i = int(np.argmin(np.abs(wl[1:] - T))) + 1
        row = f"{wl[i]:10.1f}" + "".join(f"{curves[k][i]:22.3f}" for k in names)
        r = curves["mu (deterministic)"][i] / max(curves["DUACS"][i], 1e-9)
        print(row + f"{r:12.3f}")

    json.dump({"wavelength_km": wl[1:].tolist(),
               **{k: curves[k][1:].tolist() for k in names}},
              open(out / "coherence.json", "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    st = {"DUACS": dict(color="#3cb44b", ls="--", lw=2.0),
          "mu (deterministic)": dict(color="#4363d8", lw=2.2),
          "member": dict(color="#e6194b", lw=1.8),
          "mean of 8": dict(color="#911eb4", lw=1.8)}
    fig, ax = plt.subplots(figsize=(8.5, 5.6), constrained_layout=True)
    for k in names:
        ax.semilogx(wl[1:], curves[k][1:], label=k, **st[k])
    ax.axhline(0.5, color="k", lw=0.9, ls=":", alpha=0.7)
    ax.text(wl[1], 0.51, "half the variance explained", fontsize=8, alpha=0.7)
    ax.invert_xaxis()
    ax.set_xlabel("wavelength [km]")
    ax.set_ylabel("coherence$^2$ with SWOT")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    ax.set_title(f"Is mu extracting what DUACS contains?  ({nseg} segments)\n"
                 "gap between DUACS and mu = headroom in stage 1",
                 fontsize=11, fontweight="bold")
    fig.savefig(out / "coherence_vs_scale.png", dpi=145, bbox_inches="tight")
    print(f"\nwrote {out}/coherence_vs_scale.png, {out}/coherence.json")


if __name__ == "__main__":
    main()
