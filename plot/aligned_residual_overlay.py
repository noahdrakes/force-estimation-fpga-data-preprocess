#!/usr/bin/env python3
"""Plot shifted residual traces with optional scaled force overlays from alignment debug CSV."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _x_axis(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    if "TIMESTAMP" in df.columns:
        t = pd.to_numeric(df["TIMESTAMP"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(t).any():
            t = t - np.nanmin(t)
            return t, "Time from start (s)"
    return np.arange(len(df), dtype=float), "Sample index"


def _joint_indices(df: pd.DataFrame) -> list[int]:
    indices = set()
    pattern = re.compile(r"JOINT_(\d+)_")
    for col in df.columns:
        m = pattern.match(col)
        if m:
            indices.add(int(m.group(1)))
    return sorted(indices)


def _to_float(df: pd.DataFrame, col: str) -> np.ndarray | None:
    if col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _load_force_csv(
    force_csv: Path,
    has_header: bool,
    time_col: int,
    force_cols: tuple[int, ...],
) -> pd.DataFrame:
    if has_header:
        df = pd.read_csv(force_csv)
        expected = ("FORCE_1", "FORCE_2", "FORCE_3")
        cols = [col for col in expected if col in df.columns]
        if not cols:
            raise ValueError("Force CSV must contain at least one of FORCE_1, FORCE_2, FORCE_3.")
        return df

    raw_df = pd.read_csv(force_csv, header=None)
    max_idx = raw_df.shape[1] - 1
    requested = (time_col, *force_cols)
    if any(idx < 0 or idx > max_idx for idx in requested):
        raise ValueError(f"Requested force/time columns {requested} exceed CSV width {raw_df.shape[1]}.")

    renamed = pd.DataFrame({"TIMESTAMP": raw_df.iloc[:, time_col]})
    for i, idx in enumerate(force_cols, start=1):
        renamed[f"FORCE_{i}"] = raw_df.iloc[:, idx]
    return renamed


def _scale_to_match(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    valid = np.isfinite(target) & np.isfinite(source)
    if np.count_nonzero(valid) < 2:
        return source.astype(float, copy=True)

    target_valid = target[valid].astype(float, copy=False)
    source_valid = source[valid].astype(float, copy=False)

    src_min = float(np.min(source_valid))
    src_max = float(np.max(source_valid))
    tgt_min = float(np.min(target_valid))
    tgt_max = float(np.max(target_valid))

    if np.isclose(src_max, src_min):
        out = np.full_like(source, np.nan, dtype=float)
        out[np.isfinite(source)] = 0.5 * (tgt_min + tgt_max)
        return out

    scale = (tgt_max - tgt_min) / (src_max - src_min)
    offset = tgt_min - scale * src_min
    return scale * source.astype(float, copy=False) + offset


def plot_overlay(
    debug_csv: Path,
    output_png: Path | None = None,
    force_csv: Path | None = None,
    force_has_header: bool = True,
    force_time_col: int = 0,
    force_value_cols: tuple[int, ...] = (1, 2, 3),
) -> Path:
    df = pd.read_csv(debug_csv)
    force_df = (
        _load_force_csv(force_csv, force_has_header, force_time_col, force_value_cols)
        if force_csv is not None
        else None
    )
    joints = _joint_indices(df)
    if not joints:
        raise ValueError("No JOINT_<i>_* columns found in alignment debug CSV.")

    x, x_label = _x_axis(df)
    fig, axes = plt.subplots(len(joints), 1, figsize=(12, 3.8 * len(joints)), sharex=True)
    if len(joints) == 1:
        axes = [axes]

    force_x, _ = _x_axis(force_df) if force_df is not None else (None, None)
    force_colors = {
        "FORCE_1": "tab:blue",
        "FORCE_2": "tab:orange",
        "FORCE_3": "tab:green",
    }

    for ax, joint_i in zip(axes, joints):
        shifted = _to_float(df, f"JOINT_{joint_i}_RESIDUAL_SHIFTED")

        lag_col = f"JOINT_{joint_i}_LAG_SAMPLES"
        corr_col = f"JOINT_{joint_i}_ALIGN_CORR"
        lag_val = df[lag_col].iloc[0] if lag_col in df.columns and not df.empty else np.nan
        corr_val = df[corr_col].iloc[0] if corr_col in df.columns and not df.empty else np.nan

        if shifted is not None:
            ax.plot(x, shifted, label="Residual shifted", linewidth=1.0, color="tab:red")
        if force_df is not None:
            force_col = f"FORCE_{joint_i}"
            if force_col in force_df.columns:
                force_vals = _to_float(force_df, force_col)
                scaled_force = _scale_to_match(shifted, force_vals) if shifted is not None else force_vals
                ax.plot(
                    force_x,
                    scaled_force,
                    label=f"{force_col} scaled",
                    linewidth=0.9,
                    alpha=0.8,
                    color=force_colors[force_col],
                )

        ax.set_title(f"Joint {joint_i} (lag={lag_val}, corr={corr_val:.4f})")
        ax.set_ylabel("Residual / Scaled force")
        ax.grid(alpha=0.25)

        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel(x_label)
    fig.tight_layout()

    out_path = output_png if output_png is not None else debug_csv.with_name(f"{debug_csv.stem}_overlay.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot shifted residual traces with optional scaled force overlays from alignment debug CSV."
    )
    parser.add_argument("debug_csv", type=Path, help="CSV produced by append_encoder_residuals --save-alignment-debug-csv.")
    parser.add_argument(
        "--force-csv",
        type=Path,
        default=None,
        help="Optional force CSV to overlay on every joint subplot.",
    )
    parser.add_argument(
        "--force-no-header",
        action="store_true",
        help="Interpret --force-csv as a no-header CSV. Column 0 is time by default.",
    )
    parser.add_argument(
        "--force-time-col",
        type=int,
        default=0,
        help="Time column index for a no-header force CSV.",
    )
    parser.add_argument(
        "--force-value-cols",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Force column indices for a no-header force CSV, in FORCE_1/2/3 order.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional output PNG path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = plot_overlay(
        args.debug_csv,
        args.output,
        args.force_csv,
        force_has_header=not args.force_no_header,
        force_time_col=args.force_time_col,
        force_value_cols=tuple(args.force_value_cols),
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
