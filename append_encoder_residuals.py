#!/usr/bin/env python3
"""Append encoder residual columns from *_encoderInfo.csv to a joints CSV.

Usage:
    python3 append_encoder_residuals.py <joints_csv> <encoder_info_csv> [--output <out_csv>]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def append_encoder_residuals(
    joints_csv: Path,
    encoder_info_csv: Path,
    output_csv: Path | None = None,
) -> Path:
    if not joints_csv.is_file():
        raise FileNotFoundError(f"Missing joints CSV: {joints_csv}")
    if not encoder_info_csv.is_file():
        raise FileNotFoundError(f"Missing encoder info CSV: {encoder_info_csv}")

    joints_df = pd.read_csv(joints_csv, header=None)
    encoder_df = pd.read_csv(encoder_info_csv)

    # Append only JOINT_1/2/3 residuals in this order.
    expected_residuals = [
        "JOINT_1_RESIDUAL",
        "JOINT_2_RESIDUAL",
        "JOINT_3_RESIDUAL",
    ]
    col_lookup = {str(c).upper(): c for c in encoder_df.columns}
    missing = [name for name in expected_residuals if name not in col_lookup]
    if missing:
        raise ValueError(
            f"Missing required residual columns in {encoder_info_csv}: {missing}"
        )
    residual_cols = [col_lookup[name] for name in expected_residuals]

    min_len = min(len(joints_df), len(encoder_df))
    if len(joints_df) != len(encoder_df):
        print(
            "Length mismatch: "
            f"joints={len(joints_df)}, encoder={len(encoder_df)}. Truncating to {min_len}."
        )

    if min_len == 0:
        raise ValueError("No rows available after alignment (min length is 0)")

    joints_df = joints_df.iloc[:min_len].reset_index(drop=True)
    residual_df = encoder_df.loc[: min_len - 1, residual_cols].reset_index(drop=True)

    out_df = pd.concat([joints_df, residual_df], axis=1)
    out_path = output_csv if output_csv is not None else joints_csv
    out_df.to_csv(out_path, index=False, header=False)

    print(
        f"Appended {len(residual_cols)} residual columns to {out_path}. "
        f"Rows written: {len(out_df)}"
    )
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append encoder residual columns to the end of a joints/interpolated_all_joints.csv file."
    )
    parser.add_argument("joints_csv", type=Path, help="Path to joints CSV (read with header=None).")
    parser.add_argument(
        "encoder_info_csv",
        type=Path,
        help="Path to encoder info CSV containing residual columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults to overwriting joints_csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    append_encoder_residuals(args.joints_csv, args.encoder_info_csv, args.output)


if __name__ == "__main__":
    main()
