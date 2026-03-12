#!/usr/bin/env python3
"""Append encoder residual columns from *_encoderInfo.csv to a joints CSV.

Usage:
    python3 append_encoder_residuals.py <joints_csv> <encoder_info_csv> [--output <out_csv>]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags


def _shift_array(arr: np.ndarray, lag: int) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=float)
    if lag == 0:
        out[:] = arr
    elif lag > 0:
        out[lag:] = arr[:-lag]
    else:
        k = -lag
        out[:-k] = arr[k:]
    return out


def _best_lag_for_alignment(
    residual: np.ndarray,
    reference: np.ndarray,
    max_lag: int,
) -> tuple[int, float]:
    valid = np.isfinite(residual) & np.isfinite(reference)
    if np.count_nonzero(valid) < 3:
        return 0, float("nan")

    r = residual[valid].astype(float, copy=False)
    x = reference[valid].astype(float, copy=False)
    r = r - np.mean(r)
    x = x - np.mean(x)

    c = correlate(r, x, mode="full")
    lags = correlation_lags(len(r), len(x), mode="full")
    keep = (lags >= -max_lag) & (lags <= max_lag)
    c = c[keep]
    lags = lags[keep]
    if c.size == 0:
        return 0, float("nan")

    best_idx = int(np.argmax(np.abs(c)))
    best_lag = int(lags[best_idx])

    denom = float(np.sqrt(np.sum(r * r) * np.sum(x * x)))
    best_corr = float(c[best_idx] / denom) if denom > np.finfo(float).eps else float("nan")
    return best_lag, best_corr


def _joint_index_from_residual_col(col_name: str) -> int | None:
    match = re.search(r"JOINT_(\d+)_RESIDUAL", str(col_name).upper())
    if match:
        return int(match.group(1))
    return None


def append_encoder_residuals(
    joints_csv: Path,
    encoder_info_csv: Path,
    output_csv: Path | None = None,
    save_alignment_debug_csv: Path | None = None,
    align_residuals: bool = False,
    align_reference: str = "encoder",
    align_max_lag: int = 200,
) -> Path:
    if not joints_csv.is_file():
        raise FileNotFoundError(f"Missing joints CSV: {joints_csv}")
    if not encoder_info_csv.is_file():
        raise FileNotFoundError(f"Missing encoder info CSV: {encoder_info_csv}")

    joints_df = pd.read_csv(joints_csv, header=None)
    encoder_df = pd.read_csv(encoder_info_csv)

    residual_cols = [c for c in encoder_df.columns if "residual" in str(c).lower()]
    if not residual_cols:
        raise ValueError(
            f"No residual columns found in {encoder_info_csv}. "
            "Expected at least one column containing 'residual'."
        )

    if align_max_lag < 0:
        raise ValueError(f"align_max_lag must be >= 0, got {align_max_lag}")

    residual_df_full = encoder_df.loc[:, residual_cols].copy()
    alignment_debug_cols: dict[str, np.ndarray] = {}
    lag_info: dict[str, int] = {}
    corr_info: dict[str, float] = {}
    if align_residuals:
        ref_prefix = "ENCODER_POS_" if align_reference == "encoder" else "MAPPED_POT_"
        for fallback_i, res_col in enumerate(residual_cols, start=1):
            joint_i = _joint_index_from_residual_col(str(res_col))
            idx = joint_i if joint_i is not None else fallback_i
            ref_col = f"{ref_prefix}{idx}"
            raw_residual = pd.to_numeric(encoder_df[res_col], errors="coerce").to_numpy(dtype=float)
            if ref_col not in encoder_df.columns:
                print(
                    f"Skipping alignment for {res_col}: missing reference column {ref_col}."
                )
                alignment_debug_cols[f"JOINT_{idx}_RESIDUAL_RAW"] = raw_residual
                alignment_debug_cols[f"JOINT_{idx}_RESIDUAL_SHIFTED"] = raw_residual
                lag_info[f"JOINT_{idx}_LAG_SAMPLES"] = 0
                corr_info[f"JOINT_{idx}_ALIGN_CORR"] = float("nan")
                continue

            residual = raw_residual
            reference = pd.to_numeric(encoder_df[ref_col], errors="coerce").to_numpy(dtype=float)
            lag, corr = _best_lag_for_alignment(residual, reference, align_max_lag)
            shifted = _shift_array(residual, lag)
            shifted_filled = pd.Series(shifted).bfill().ffill().to_numpy(dtype=float)
            residual_df_full[res_col] = shifted_filled
            lag_info[f"JOINT_{idx}_LAG_SAMPLES"] = lag
            corr_info[f"JOINT_{idx}_ALIGN_CORR"] = corr
            alignment_debug_cols[f"JOINT_{idx}_RESIDUAL_RAW"] = residual
            alignment_debug_cols[f"JOINT_{idx}_RESIDUAL_SHIFTED"] = shifted_filled
            if ref_col in encoder_df.columns:
                alignment_debug_cols[f"JOINT_{idx}_ALIGN_REFERENCE"] = pd.to_numeric(
                    encoder_df[ref_col], errors="coerce"
                ).to_numpy(dtype=float)
            enc_col = f"ENCODER_POS_{idx}"
            pot_col = f"MAPPED_POT_{idx}"
            if enc_col in encoder_df.columns:
                alignment_debug_cols[f"JOINT_{idx}_ENCODER_POS"] = pd.to_numeric(
                    encoder_df[enc_col], errors="coerce"
                ).to_numpy(dtype=float)
            if pot_col in encoder_df.columns:
                alignment_debug_cols[f"JOINT_{idx}_MAPPED_POT"] = pd.to_numeric(
                    encoder_df[pot_col], errors="coerce"
                ).to_numpy(dtype=float)
            print(
                f"{res_col} alignment: lag={lag} samples, corr={corr:.6f}, reference={ref_col}"
            )

    min_len = min(len(joints_df), len(encoder_df))
    if len(joints_df) != len(encoder_df):
        print(
            "Length mismatch: "
            f"joints={len(joints_df)}, encoder={len(encoder_df)}. Truncating to {min_len}."
        )

    if min_len == 0:
        raise ValueError("No rows available after alignment (min length is 0)")

    joints_df = joints_df.iloc[:min_len].reset_index(drop=True)
    residual_df = residual_df_full.iloc[:min_len].reset_index(drop=True)

    out_df = pd.concat([joints_df, residual_df], axis=1)
    out_path = output_csv if output_csv is not None else joints_csv
    out_df.to_csv(out_path, index=False, header=False)

    if save_alignment_debug_csv is not None:
        debug_df = pd.DataFrame()
        if "TIMESTAMP" in encoder_df.columns:
            debug_df["TIMESTAMP"] = pd.to_numeric(encoder_df["TIMESTAMP"], errors="coerce")
        else:
            debug_df["SAMPLE_INDEX"] = np.arange(len(encoder_df), dtype=int)

        for key, values in alignment_debug_cols.items():
            debug_df[key] = values

        for key, value in lag_info.items():
            debug_df[key] = value
        for key, value in corr_info.items():
            debug_df[key] = value

        debug_df = debug_df.iloc[:min_len].reset_index(drop=True)
        save_alignment_debug_csv.parent.mkdir(parents=True, exist_ok=True)
        debug_df.to_csv(save_alignment_debug_csv, index=False)
        print(f"Saved alignment debug CSV: {save_alignment_debug_csv}")

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
    parser.add_argument(
        "--save-alignment-debug-csv",
        type=Path,
        default=None,
        help=(
            "Optional path to save raw/shifted residual alignment debug data, "
            "including encoder and mapped pot traces when present."
        ),
    )
    parser.add_argument(
        "--no-align-residuals",
        action="store_true",
        help="Disable residual alignment before appending.",
    )
    parser.add_argument(
        "--align-reference",
        type=str,
        default="encoder",
        choices=["encoder", "mapped_pot"],
        help="Signal used to align each residual column.",
    )
    parser.add_argument(
        "--align-max-lag",
        type=int,
        default=300,
        help="Max lag (in samples) searched in both directions for alignment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    append_encoder_residuals(
        args.joints_csv,
        args.encoder_info_csv,
        args.output,
        align_residuals=False,
        align_reference=args.align_reference,
        align_max_lag=args.align_max_lag,
    )


if __name__ == "__main__":
    main()
