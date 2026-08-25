#!/usr/bin/env python3
"""
main.py
-------
CLI entry point for the hyperspectral field crop analysis pipeline.

Quick start
-----------
# Process local folder
python main.py --local-folder ./data

# Process from GitHub repo
python main.py --github-repo owner/reponame --github-folder data/2024

# Override method and number of classes
python main.py --local-folder ./data --method kmeans --n-classes 8
python main.py --local-folder ./data --method hybrid --n-classes 9

# Use supervised classification (requires labelled pixels CSV)
python main.py --local-folder ./data --method supervised --labels labels.csv

# Test with first 2 files only
python main.py --local-folder ./data --limit 2

# Custom config file
python main.py --config my_config.yaml --local-folder ./data
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml


# ------------------------------------------------------------------ #
# Logging setup
# ------------------------------------------------------------------ #

def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s  %(levelname)-7s  %(message)s"
    # Windows 콘솔은 UTF-8로 강제 설정
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.stream.reconfigure(encoding="utf-8", errors="replace") \
        if hasattr(stream_handler.stream, "reconfigure") else None
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            stream_handler,
            logging.FileHandler("pipeline.log", encoding="utf-8"),
        ],
    )


# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #

def load_config(config_path: str) -> dict:
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        logging.warning(f"Config file '{config_path}' not found – using built-in defaults")
        return {}
    with open(cfg_file, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def merge_cli_into_config(cfg: dict, args: argparse.Namespace) -> dict:
    """Allow CLI flags to override config.yaml values."""
    data_cfg = cfg.setdefault("data", {})
    clf_cfg  = cfg.setdefault("classification", {})
    out_cfg  = cfg.setdefault("output", {})

    if args.local_folder:
        data_cfg["local_folder"] = args.local_folder

    if args.github_repo:
        gh = data_cfg.setdefault("github", {})
        gh["repo"] = args.github_repo
        if args.github_folder:
            gh["folder"] = args.github_folder
        if args.github_token:
            gh["token"] = args.github_token

    if args.output_dir:
        out_cfg["dir"] = args.output_dir

    if args.method:
        clf_cfg["method"] = args.method

    if args.n_classes is not None:
        n = args.n_classes
        # kmeans: use directly
        clf_cfg.setdefault("kmeans", {})["n_clusters"] = n
        # hybrid: distribute evenly across the 3 main segments
        base = max(1, n // 3)
        r    = n % 3
        clf_cfg.setdefault("hybrid", {}).update({
            "n_clusters_sunlit": base + (1 if r >= 1 else 0),
            "n_clusters_shadow": base + (1 if r >= 2 else 0),
            "n_clusters_soil":   base,
        })
        logger.info(
            f"n-classes={n} → "
            f"sunlit={base+(1 if r>=1 else 0)}, "
            f"shadow={base+(1 if r>=2 else 0)}, "
            f"soil={base}"
        ) if False else None  # logged in pipeline

    if args.ndvi_threshold is not None:
        clf_cfg.setdefault("hybrid", {})["ndvi_threshold"] = args.ndvi_threshold

    if args.brightness_threshold is not None:
        clf_cfg.setdefault("hybrid", {})["brightness_threshold"] = args.brightness_threshold

    return cfg


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hyperspectral field crop analysis – auto spectrum extraction",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # --- Data sources ---
    src = parser.add_argument_group("Data sources")
    src.add_argument(
        "--local-folder", metavar="DIR",
        help="Local folder containing hyperspectral files",
    )
    src.add_argument(
        "--github-repo", metavar="OWNER/REPO",
        help="GitHub repository (e.g. myuser/field-data)",
    )
    src.add_argument(
        "--github-folder", metavar="PATH", default="",
        help="Sub-folder inside the GitHub repo (default: root)",
    )
    src.add_argument(
        "--github-token", metavar="TOKEN",
        help="GitHub personal access token (or set GITHUB_TOKEN env var)",
    )

    # --- Classification ---
    clf = parser.add_argument_group("Classification")
    clf.add_argument(
        "--method",
        choices=["hybrid", "kmeans", "sam", "supervised", "autoencoder", "cnn"],
        help=(
            "Classification method (overrides config.yaml)\n"
            "  hybrid      : NDVI+brightness rules -> K-means (no labels)\n"
            "  kmeans      : PCA + K-means (no labels)\n"
            "  sam         : Spectral Angle Mapper - brightness-invariant (no labels OR --labels)\n"
            "  autoencoder : Spectral AE -> latent K-means (no labels, deep)\n"
            "  supervised  : Random Forest (needs --labels CSV)\n"
            "  cnn         : 1D-CNN (needs --labels CSV, deep)\n"
        ),
    )
    clf.add_argument(
        "--n-classes", type=int, metavar="N",
        help=(
            "Total number of output classes/clusters.\n"
            "  hybrid : distributed across sunlit/shadow/soil segments\n"
            "           (e.g. 9 → 3 sub-clusters each)\n"
            "  kmeans : used directly as n_clusters"
        ),
    )
    clf.add_argument(
        "--ndvi-threshold", type=float, metavar="FLOAT",
        help="NDVI threshold for vegetation detection (hybrid method, default 0.15)",
    )
    clf.add_argument(
        "--brightness-threshold", type=float, metavar="FLOAT",
        help="Brightness threshold for shadow detection (hybrid method, default 0.08)",
    )
    clf.add_argument(
        "--labels", dest="labels_csv", metavar="CSV",
        help="Labelled pixels CSV (row,col,class_id) for supervised method",
    )

    # --- Output / misc ---
    misc = parser.add_argument_group("Output / misc")
    misc.add_argument(
        "--output-dir", metavar="DIR", default=None,
        help="Output directory (default: ./output)",
    )
    misc.add_argument(
        "--config", metavar="YAML", default="config.yaml",
        help="Path to config YAML file (default: config.yaml)",
    )
    misc.add_argument(
        "--limit", type=int, metavar="N",
        help="Process at most N files (useful for quick tests)",
    )
    misc.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging",
    )

    # Subcommand: list – just discover files, don't process
    sub = parser.add_subparsers(dest="command")
    lst = sub.add_parser("list", help="List discoverable hyperspectral files and exit")
    lst.add_argument("--local-folder", metavar="DIR")
    lst.add_argument("--github-repo",  metavar="OWNER/REPO")
    lst.add_argument("--github-folder", metavar="PATH", default="")
    lst.add_argument("--github-token",  metavar="TOKEN")
    lst.add_argument("--config", metavar="YAML", default="config.yaml")

    args = parser.parse_args()
    setup_logging(getattr(args, "verbose", False))
    log = logging.getLogger(__name__)

    cfg = load_config(args.config)
    cfg = merge_cli_into_config(cfg, args)

    # ---- Subcommand: list ----
    if args.command == "list":
        from src.data_loader import HyperspectralLoader
        loader = HyperspectralLoader(cfg.get("data", {}))
        data_cfg = cfg.get("data", {})
        folder = data_cfg.get("local_folder")
        if folder:
            files = loader.list_local_files(folder)
            print(f"\nLocal files in '{folder}':")
            for f in files:
                print(f"  {f}")
        gh = data_cfg.get("github", {})
        if gh.get("repo"):
            files = loader.list_github_files(
                gh["repo"], gh.get("folder", ""), gh.get("token")
            )
            print(f"\nGitHub files in '{gh['repo']}':")
            for f in files:
                print(f"  {f}")
        return

    # ---- Normal run ----
    from src.pipeline import Pipeline

    log.info("Hyperspectral pipeline starting")
    log.info(f"Config: {args.config}")
    clf_info = cfg.get("classification", {})
    method   = clf_info.get("method", "hybrid")
    log.info(f"Method: {method}")

    # ── 클래스 수 입력 ──────────────────────────────────────────
    # --n-classes 가 주어지지 않으면 대화형으로 묻습니다.
    if args.n_classes is None:
        default_n = clf_info.get("kmeans", {}).get("n_clusters", 6)
        try:
            raw = input(
                f"\n클래스(클러스터) 수를 입력하세요 "
                f"[기본값 {default_n}]: "
            ).strip()
            n_classes = int(raw) if raw else default_n
        except (ValueError, EOFError):
            n_classes = default_n
            print(f"  잘못된 입력 → 기본값 {default_n} 사용")

        # config에 반영
        base = max(1, n_classes // 3)
        r_   = n_classes % 3
        clf_cfg = cfg.setdefault("classification", {})
        clf_cfg.setdefault("kmeans", {})["n_clusters"] = n_classes
        clf_cfg.setdefault("hybrid", {}).update({
            "n_clusters_sunlit": base + (1 if r_ >= 1 else 0),
            "n_clusters_shadow": base + (1 if r_ >= 2 else 0),
            "n_clusters_soil":   base,
        })
        print(
            f"  -> 클래스 수: {n_classes}  "
            f"(kmeans={n_classes}, "
            f"hybrid: sunlit={base+(1 if r_>=1 else 0)}, "
            f"shadow={base+(1 if r_>=2 else 0)}, "
            f"soil={base})\n"
        )
    else:
        n_classes = args.n_classes
    log.info(f"n-classes: {n_classes}")
    # ────────────────────────────────────────────────────────────

    pipeline = Pipeline(cfg)
    pipeline.run(
        labels_csv=getattr(args, "labels_csv", None),
        file_limit=getattr(args, "limit", None),
    )


if __name__ == "__main__":
    main()
