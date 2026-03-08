#!/usr/bin/env python3
"""Export raw sim/ADC datasets into normalized training schemas."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.dataset_export import (
    ADC_COMPATIBLE_SCHEMA,
    CANONICAL_TRAINING_SCHEMA,
    export_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export raw simulator or ADC datasets into normalized schemas"
    )
    parser.add_argument(
        "source_path",
        help="Path to a sim raw dataset dir or ADC LeRobot dataset dir",
    )
    parser.add_argument(
        "--source-type",
        required=True,
        choices=["sim_raw", "adc_raw"],
        help="Source dataset type",
    )
    parser.add_argument(
        "--schema",
        default=CANONICAL_TRAINING_SCHEMA,
        choices=[ADC_COMPATIBLE_SCHEMA, CANONICAL_TRAINING_SCHEMA],
        help="Target export schema",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the exported dataset will be written",
    )
    parser.add_argument(
        "--robot",
        default=None,
        help="Canonical robot name override (for adc_raw or ambiguous sources)",
    )
    parser.add_argument(
        "--adc-robot-config",
        default=None,
        help="Path to ADC robot config YAML (adc_raw only)",
    )
    parser.add_argument(
        "--no-link-images",
        action="store_true",
        help="Do not link source image/video artifacts into the export directory",
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

    export_root = export_dataset(
        source_path=args.source_path,
        source_type=args.source_type,
        schema=args.schema,
        output_dir=args.output_dir,
        robot_name=args.robot,
        adc_robot_config=args.adc_robot_config,
        link_images=not args.no_link_images,
    )
    print(export_root)


if __name__ == "__main__":
    main()
