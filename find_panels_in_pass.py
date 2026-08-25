#!/usr/bin/env python3
"""
find_panels_in_pass.py
----------------------
Search a HySpex pass for calibration panels without loading the whole cube.

Panels are usually laid out at the start or the end of a pass, so this reads a
block of consecutive lines from each end. BIL stores whole lines contiguously,
which makes that a fast sequential read — striding a single band across the
file instead would walk all 24 GB per band.

Candidates are ranked by brightness among regions that are bright, compact and
spectrally flat once the sensor's quantum-efficiency curve is divided out.

Usage
-----
    python find_panels_in_pass.py Z:/.../vnir.bil.hdr
    python find_panels_in_pass.py Z:/.../vnir.bil.hdr --h5 Z:/.../scene.hyspex.h5
    python find_panels_in_pass.py Z:/.../vnir.bil.hdr --lines 400 --whole

Feed the reported line/sample ranges into the 패널 보정 tab, or use them as
explicit boxes with src.radiometry.reflectance_from_reference().
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("hdr", help="ENVI header of the pass (e.g. vnir.bil.hdr)")
    p.add_argument("--h5", default=None,
                   help="HySpex .hyspex.h5 for the quantum-efficiency curve. "
                        "Without it, flatness is far less reliable.")
    p.add_argument("--lines", type=int, default=250,
                   help="Lines to read from each end (default 250)")
    p.add_argument("--whole", action="store_true",
                   help="Scan the entire pass instead of just the two ends "
                        "(slow: reads the whole file)")
    p.add_argument("--sample-step", type=int, default=2,
                   help="Across-track decimation while searching (default 2)")
    p.add_argument("--min-pixels", type=int, default=80,
                   help="Smallest accepted blob (default 80)")
    p.add_argument("--top", type=int, default=6,
                   help="Candidates to report per block (default 6)")
    p.add_argument("--out", default="./output/panel_candidates",
                   help="Prefix for the .npz/.png written out")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import spectral
        from scipy import ndimage
    except ImportError as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        return 2

    hdr = Path(args.hdr)
    if not hdr.exists():
        print(f"Header not found: {hdr}", file=sys.stderr)
        print("If this is a network path, check the drive is connected.",
              file=sys.stderr)
        return 2

    img = spectral.open_image(str(hdr))
    wl = np.array(img.bands.centers) if img.bands.centers else None
    mm = img.open_memmap(interleave="source")        # (lines, bands, samples)
    L, B, S = mm.shape
    print(f"cube {mm.shape}  (lines, bands, samples)", flush=True)

    qe = None
    if args.h5:
        import h5py
        with h5py.File(args.h5, "r") as f:
            qe = f["vnir/intrinsics/quantum_efficiency"][:].astype(np.float32)
        if len(qe) != B:
            print(f"  QE has {len(qe)} bands but cube has {B}; ignoring",
                  file=sys.stderr)
            qe = None
        else:
            print(f"  QE loaded ({qe.min():.5f}-{qe.max():.5f})", flush=True)
    else:
        print("  No --h5 given: flatness is computed on raw DN and will look "
              "poor even for a neutral panel.", flush=True)

    nl = min(args.lines, L)
    blocks = ({"whole": (0, L)} if args.whole
              else {"head": (0, nl), "tail": (max(0, L - nl), L)})

    step = max(1, args.sample_step)
    found: list[tuple[str, dict]] = []

    for name, (l0, l1) in blocks.items():
        print(f"\n=== {name}: lines {l0}-{l1} ===", flush=True)
        sub = np.asarray(mm[l0:l1, :, ::step], dtype=np.float32)
        cube = np.transpose(sub, (0, 2, 1))          # (nl, s', B)
        del sub

        work = cube / qe if qe is not None else cube
        bright = work.mean(axis=2)
        cv = work.std(axis=2) / np.maximum(bright, 1e-9)
        print(f"  block {cube.shape}  DN {cube.min():.0f}-{cube.max():.0f}",
              flush=True)
        print(f"  brightness p50={np.percentile(bright, 50):.1f} "
              f"p99={np.percentile(bright, 99):.1f} max={bright.max():.1f}",
              flush=True)

        mask = (bright > np.percentile(bright, 97)) & \
               (cv < np.percentile(cv, 25))
        lab, n = ndimage.label(mask)

        cand: list[dict] = []
        for k in range(1, n + 1):
            m = lab == k
            npx = int(m.sum())
            if npx < args.min_pixels:
                continue
            rr, cc = np.where(m)
            r0, r1 = int(rr.min()), int(rr.max()) + 1
            c0, c1 = int(cc.min()), int(cc.max()) + 1
            # A panel fills most of its bounding box; reject stringy blobs.
            if npx < 0.5 * (r1 - r0) * (c1 - c0):
                continue
            spec = cube[r0:r1, c0:c1, :].reshape(-1, B).mean(axis=0)
            sq = spec / qe if qe is not None else spec
            cand.append({
                "lines": (l0 + r0, l0 + r1),
                "samples": (c0 * step, c1 * step),
                "n_px": npx,
                "bright": float(bright[m].mean()),
                "flatness": float(np.std(sq / sq.mean())),
                "spectrum": spec,
            })

        cand.sort(key=lambda d: d["bright"], reverse=True)
        print(f"  candidates: {len(cand)}", flush=True)
        for i, c in enumerate(cand[:args.top], 1):
            print(f"    #{i} lines {c['lines'][0]}-{c['lines'][1]}  "
                  f"samples {c['samples'][0]}-{c['samples'][1]}  "
                  f"px={c['n_px']:,}  bright={c['bright']:.1f}  "
                  f"flat={c['flatness']:.3f}", flush=True)
        found += [(name, c) for c in cand[:args.top]]
        del cube, work

    if not found:
        print("\nNo panel-like regions found. Try --whole, a larger --lines, "
              "or a smaller --min-pixels.", flush=True)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out.with_suffix(".npz"),
             wl=(wl if wl is not None else np.arange(B)),
             specs=np.array([c["spectrum"] for _, c in found]),
             lines=np.array([c["lines"] for _, c in found]),
             samples=np.array([c["samples"] for _, c in found]),
             flatness=np.array([c["flatness"] for _, c in found]))
    print(f"\nsaved {out.with_suffix('.npz')}", flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = wl if wl is not None else np.arange(B)
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        for nm, c in found[:8]:
            ax.plot(x, c["spectrum"], lw=1.5,
                    label=f"{nm} L{c['lines'][0]} flat={c['flatness']:.2f}")
        ax.set_xlabel("Wavelength (nm)" if wl is not None else "Band")
        ax.set_ylabel("raw DN")
        ax.set_title(f"Panel candidates — {hdr.name}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out.with_suffix(".png"), dpi=130, bbox_inches="tight")
        print(f"saved {out.with_suffix('.png')}", flush=True)
    except Exception as e:                       # plotting is a convenience
        print(f"(plot skipped: {e})", flush=True)

    print("\nA true grey-scale set shows several candidates at clearly "
          "different brightness with similar, low flatness.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
