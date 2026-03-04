#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import filtfilt, firwin


def _infer_sampling_rate(timestamps: pd.Series) -> float:
    t = pd.to_numeric(timestamps, errors="coerce").to_numpy(dtype=float)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError("Cannot infer sampling rate from TIMESTAMP.")
    return 1.0 / float(np.median(dt))


def _kaiser_lowpass_filter(
    signal: pd.Series,
    fs: float,
    cutoff_hz: float,
    order: int = 30,
    beta: float = 8.6,
) -> np.ndarray:
    x = pd.to_numeric(signal, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x)
    if np.count_nonzero(valid) < 8:
        return x

    if cutoff_hz <= 0.0:
        raise ValueError(f"Cutoff frequency must be > 0 Hz, got {cutoff_hz}.")
    nyquist = 0.5 * fs
    if cutoff_hz >= nyquist:
        raise ValueError(
            f"Cutoff {cutoff_hz:.3f} Hz must be below Nyquist ({nyquist:.3f} Hz)."
        )

    taps = firwin(
        numtaps=int(order) + 1,
        cutoff=float(cutoff_hz),
        fs=fs,
        pass_zero="lowpass",
        window=("kaiser", float(beta)),
    )
    y = x.copy()
    y_valid = filtfilt(taps, [1.0], x[valid])
    y[valid] = y_valid
    return y


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot encoder vs pot signals from *_potEncoder.csv")
    parser.add_argument("csv", type=str, help="Path to *_potEncoder.csv")
    parser.add_argument(
        "--residual-kaiser-cutoff",
        type=float,
        default=None,
        help="Apply a Kaiser low-pass FIR filter to residual traces using this cutoff frequency (Hz).",
    )
    parser.add_argument(
        "--kaiser-beta",
        type=float,
        default=8.6,
        help="Kaiser window beta parameter (higher = stronger stop-band attenuation).",
    )
    parser.add_argument(
        "--kaiser-order",
        type=int,
        default=30,
        help="FIR filter order for residual Kaiser low-pass (numtaps = order + 1).",
    )
    parser.add_argument(
        "--print-filter-delta",
        action="store_true",
        help="Print per-joint residual change stats after notch filtering.",
    )
    parser.add_argument(
        "--save-filtered-residual-csv",
        nargs="?",
        const="",
        default=None,
        help=(
            "Save plotted residual traces to CSV. "
            "If a path is provided, write there; otherwise uses "
            "<input_stem>_encoder_residual_filtered.csv next to input."
        ),
    )
    args = parser.parse_args()
    if args.kaiser_order < 1:
        raise ValueError("--kaiser-order must be >= 1.")

    csv_path = Path(args.csv).expanduser().resolve()
    df = pd.read_csv(csv_path)

    t = pd.to_numeric(df["TIMESTAMP"], errors="coerce")
    t = t - t.iloc[0]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    pairs = [("POT_3", "ENCODER_POS_1", "ORIGINAL_ENCODER_POS_1"),
             ("POT_4", "ENCODER_POS_2", "ORIGINAL_ENCODER_POS_2"),
             ("POT_5", "ENCODER_POS_3", "ORIGINAL_ENCODER_POS_3")]

    for i, (pot, enc, enc_raw) in enumerate(pairs, start=1):
        ax = axes[i - 1]
        ax.plot(t, df[pot], label=pot, linewidth=0.9)
        ax.plot(t, df[enc], label=f"{enc} (mapped)", linewidth=0.9)
        ax.plot(t, df[enc_raw], label=f"{enc_raw} (raw)", linewidth=0.9, alpha=0.7)
        ax.set_title(f"Joint {i}")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time from start")
    fig.tight_layout()

    out_png = csv_path.with_name(f"{csv_path.stem}_encoder_vs_pot.png")
    fig.savefig(out_png, dpi=200)
    print(f"Saved {out_png}")

    fig_res, axes_res = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fs = None
    cutoff_hz = args.residual_kaiser_cutoff
    residual_df = pd.DataFrame({"TIMESTAMP": pd.to_numeric(df["TIMESTAMP"], errors="coerce"), "TIME_FROM_START": t})
    residual_series = []
    if cutoff_hz is not None:
        fs = _infer_sampling_rate(df["TIMESTAMP"])
        print(
            f"Applying Kaiser FIR low-pass to residuals "
            f"(order={args.kaiser_order}, cutoff={cutoff_hz:g} Hz, "
            f"beta={args.kaiser_beta}, fs={fs:.3f} Hz)."
        )
    for i, (_pot, enc, enc_raw) in enumerate(pairs, start=1):
        ax = axes_res[i - 1]
        residual = pd.to_numeric(df[enc], errors="coerce") - pd.to_numeric(df[enc_raw], errors="coerce")
        residual_to_plot = residual
        if cutoff_hz is not None:
            residual_to_plot = _kaiser_lowpass_filter(
                residual,
                fs=fs,
                cutoff_hz=cutoff_hz,
                order=args.kaiser_order,
                beta=args.kaiser_beta,
            )
            if args.print_filter_delta:
                raw = pd.to_numeric(residual, errors="coerce").to_numpy(dtype=float)
                filt = np.asarray(residual_to_plot, dtype=float)
                valid = np.isfinite(raw) & np.isfinite(filt)
                if np.any(valid):
                    delta = filt[valid] - raw[valid]
                    mean_abs_delta = float(np.mean(np.abs(delta)))
                    max_abs_delta = float(np.max(np.abs(delta)))
                    raw_rms = float(np.sqrt(np.mean(raw[valid] ** 2)))
                    filt_rms = float(np.sqrt(np.mean(filt[valid] ** 2)))
                    print(
                        f"Joint {i}: mean|delta|={mean_abs_delta:.6g}, "
                        f"max|delta|={max_abs_delta:.6g}, rms(raw)={raw_rms:.6g}, rms(filt)={filt_rms:.6g}"
                    )
        residual_to_plot = np.asarray(residual_to_plot, dtype=float)
        residual_series.append(residual_to_plot)
        residual_df[f"JOINT_{i}_RESIDUAL"] = residual_to_plot
        ax.plot(t, residual_to_plot, linewidth=0.9)
        ax.set_title(f"Joint {i} residual: {enc} - {enc_raw}")
        ax.set_ylabel("Residual")
        ax.grid(alpha=0.25)

    axes_res[-1].set_xlabel("Time from start")
    fig_res.tight_layout()
    out_res_png = csv_path.with_name(f"{csv_path.stem}_encoder_residual.png")
    fig_res.savefig(out_res_png, dpi=200)
    print(f"Saved {out_res_png}")
    fig_overlay, axes_overlay = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for i, (pot, enc, enc_raw) in enumerate(pairs, start=1):
        ax = axes_overlay[i - 1]
        ax.plot(t, df[pot], label=pot, linewidth=0.9)
        ax.plot(t, df[enc], label=f"{enc} (mapped)", linewidth=0.9)
        ax.plot(t, df[enc_raw], label=f"{enc_raw} (raw)", linewidth=0.9, alpha=0.7)
        ax2 = ax.twinx()
        ax2.plot(t, residual_series[i - 1], color="green", linewidth=0.9, label="Residual")
        ax.set_title(f"Joint {i}: encoder/pot with residual")
        ax.set_ylabel("Encoder/POT")
        ax2.set_ylabel("Residual")
        ax.grid(alpha=0.25)
        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=8)
    axes_overlay[-1].set_xlabel("Time from start")
    fig_overlay.tight_layout()
    out_overlay_png = csv_path.with_name(f"{csv_path.stem}_encoder_pot_residual_overlay.png")
    fig_overlay.savefig(out_overlay_png, dpi=200)
    print(f"Saved {out_overlay_png}")
    if args.save_filtered_residual_csv is not None:
        if args.save_filtered_residual_csv == "":
            out_res_csv = csv_path.with_name(f"{csv_path.stem}_encoder_residual_filtered.csv")
        else:
            out_res_csv = Path(args.save_filtered_residual_csv).expanduser().resolve()
        residual_df.to_csv(out_res_csv, index=False)
        print(f"Saved {out_res_csv}")


if __name__ == "__main__":
    main()
