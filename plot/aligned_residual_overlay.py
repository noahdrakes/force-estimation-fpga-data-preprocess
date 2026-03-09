#!/usr/bin/env python3
"""Plot encoder/mapped pot with raw and shifted residual traces from alignment debug CSV."""

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


def plot_overlay(debug_csv: Path, output_png: Path | None = None) -> Path:
    df = pd.read_csv(debug_csv)
    joints = _joint_indices(df)
    if not joints:
        raise ValueError("No JOINT_<i>_* columns found in alignment debug CSV.")

    x, x_label = _x_axis(df)
    fig, axes = plt.subplots(len(joints), 1, figsize=(12, 3.8 * len(joints)), sharex=True)
    if len(joints) == 1:
        axes = [axes]

    for ax, joint_i in zip(axes, joints):
        enc = _to_float(df, f"JOINT_{joint_i}_ENCODER_POS")
        pot = _to_float(df, f"JOINT_{joint_i}_MAPPED_POT")
        ref = _to_float(df, f"JOINT_{joint_i}_ALIGN_REFERENCE")
        raw = _to_float(df, f"JOINT_{joint_i}_RESIDUAL_RAW")
        shifted = _to_float(df, f"JOINT_{joint_i}_RESIDUAL_SHIFTED")

        lag_col = f"JOINT_{joint_i}_LAG_SAMPLES"
        corr_col = f"JOINT_{joint_i}_ALIGN_CORR"
        lag_val = df[lag_col].iloc[0] if lag_col in df.columns and not df.empty else np.nan
        corr_val = df[corr_col].iloc[0] if corr_col in df.columns and not df.empty else np.nan

        if enc is not None:
            ax.plot(x, enc, label="Encoder", linewidth=1.0)
        if pot is not None:
            ax.plot(x, pot, label="Mapped pot", linewidth=1.0, alpha=0.9)
        # if ref is not None:
        #     ax.plot(x, ref, label="Align reference", linewidth=0.9, alpha=0.65)

        ax2 = ax.twinx()
        if raw is not None:
            ax2.plot(x, raw, label="Residual raw", linewidth=0.9, color="tab:green", alpha=0.6)
        # if shifted is not None:
        #     ax2.plot(x, shifted, label="Residual shifted", linewidth=0.9, color="tab:red")

        ax.set_title(f"Joint {joint_i} (lag={lag_val}, corr={corr_val:.4f})")
        ax.set_ylabel("Encoder/Pot")
        ax2.set_ylabel("Residual")
        ax.grid(alpha=0.25)

        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=8)

    axes[-1].set_xlabel(x_label)
    fig.tight_layout()

    out_path = output_png if output_png is not None else debug_csv.with_name(f"{debug_csv.stem}_overlay.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot encoder/pot with raw and shifted residual traces from alignment debug CSV."
    )
    parser.add_argument("debug_csv", type=Path, help="CSV produced by append_encoder_residuals --save-alignment-debug-csv.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output PNG path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = plot_overlay(args.debug_csv, args.output)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
