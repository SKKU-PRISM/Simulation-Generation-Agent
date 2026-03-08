#!/usr/bin/env python3
"""Preprocess an exported dataset into per-robot training manifests."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.dataset_preprocess import preprocess_exported_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess an adc_compatible export into per-robot training manifests"
    )
    parser.add_argument(
        "export_dir",
        help="Path to an adc_compatible exported dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the preprocessing manifests will be written",
    )
    parser.add_argument(
        "--camera-names",
        default="",
        help="Comma-separated camera names to keep. Default: all available cameras in the export",
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="Keep only successful episodes when generating manifests",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Episode-level train split ratio (default: 0.9)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic episode split (default: 42)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    camera_names = [name.strip() for name in args.camera_names.split(",") if name.strip()]
    preprocess_root = preprocess_exported_dataset(
        export_dir=args.export_dir,
        output_dir=args.output_dir,
        camera_names=camera_names or None,
        success_only=args.success_only,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    print(preprocess_root)


if __name__ == "__main__":
    main()
