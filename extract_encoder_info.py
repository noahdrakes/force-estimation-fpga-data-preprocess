#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import filtfilt, firwin


def _fit_linear_map(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    x_num = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_num = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(x_num) & np.isfinite(y_num)
    if np.count_nonzero(valid) < 2:
        return 1.0, 0.0

    x_valid = x_num[valid]
    y_valid = y_num[valid]
    x_mean = x_valid.mean()
    y_mean = y_valid.mean()
    denom = np.sum((x_valid - x_mean) ** 2)
    if denom <= np.finfo(float).eps:
        return 1.0, float(y_mean - x_mean)

    slope = float(np.sum((x_valid - x_mean) * (y_valid - y_mean)) / denom)
    intercept = float(y_mean - slope * x_mean)
    return slope, intercept


def _infer_sampling_rate(timestamps: pd.Series) -> float:
    t = pd.to_numeric(timestamps, errors="coerce").to_numpy(dtype=float)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError("Cannot infer sampling rate from TIMESTAMP.")
    return 1.0 / float(np.median(dt))


def _design_fir_filter(filter_type: str, fs: float, cutoff_hz: float, order: int) -> np.ndarray:
    fC_norm = float(cutoff_hz) / (float(fs) / 2.0)
    if fC_norm <= 0 or fC_norm >= 1:
        raise ValueError(
            f"cutoff_hz must be between 0 and Nyquist ({fs/2:.6g} Hz); got {cutoff_hz}."
        )
    if filter_type == "kaiser":
        return firwin(int(order) + 1, fC_norm, window=("kaiser", 3.5))
    if filter_type == "hamming":
        return firwin(int(order) + 1, fC_norm, window="hamming")
    if filter_type == "chebyshev":
        return firwin(int(order) + 1, fC_norm, window=("chebwin", 40))
    raise ValueError(f"Unknown filter type: {filter_type}")


def _apply_filter_to_extracted_df(df: pd.DataFrame, fir_coeffs: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    value_cols = [c for c in out.columns if c != "TIMESTAMP"]
    arr = out[value_cols].to_numpy(dtype=float)
    valid = np.all(np.isfinite(arr), axis=1)
    if np.count_nonzero(valid) > max(8, len(fir_coeffs)):
        arr[valid] = filtfilt(fir_coeffs, [1.0], arr[valid], axis=0)
    out[value_cols] = arr
    return out


def _downsample_df(
    df: pd.DataFrame,
    original_freq: float,
    target_freq: float,
    use_moving_average: bool,
) -> pd.DataFrame:
    window_size = int(float(original_freq) / float(target_freq))
    if window_size < 1:
        raise ValueError("Target frequency must be lower than or equal to original frequency.")

    timestamps = df.iloc[:, 0]
    data = df.iloc[:, 1:]
    if use_moving_average:
        n_windows = len(df) // window_size
        ts_ds = timestamps.iloc[: n_windows * window_size].groupby(
            timestamps.index[: n_windows * window_size] // window_size
        ).mean().reset_index(drop=True)
        data_ds = data.iloc[: n_windows * window_size].groupby(
            data.index[: n_windows * window_size] // window_size
        ).mean().reset_index(drop=True)
    else:
        ts_ds = timestamps.iloc[::window_size].reset_index(drop=True)
        data_ds = data.iloc[::window_size].reset_index(drop=True)
    return pd.concat([ts_ds, data_ds], axis=1)


def extract_encoder_info(
    input_csv: str,
    output_csv: str,
    plot: bool = False,
    pot_filter: bool = False,
    pot_filter_cutoff_hz: float = 30.0,
    pot_filter_order: int = 30,
    pot_filter_type: str = "kaiser",
    pot_downsample: bool = False,
    pot_downsample_freq: float | None = None,
    pot_original_freq: float | None = None,
    pot_downsample_moving_average: bool = True,
) -> Path:
    input_path = Path(input_csv).expanduser().resolve()
    output_path = Path(output_csv).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)

    required = ["TIMESTAMP", "POT_3", "POT_4", "POT_5"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    pairs = [("POT_3", 1.0), ("POT_4", 1.0), ("POT_5", -1.0)]
    t = pd.to_numeric(df["TIMESTAMP"], errors="coerce")
    out = pd.DataFrame({"TIMESTAMP": t})

    for i, (pot_col, sign) in enumerate(pairs, start=1):
        raw_enc_col = f"ORIGINAL_ENCODER_POS_{i}"
        fallback_enc_col = f"ENCODER_POS_{i}"
        enc_col = raw_enc_col if raw_enc_col in df.columns else fallback_enc_col
        if enc_col not in df.columns:
            raise ValueError(
                f"Missing encoder column for joint {i}: expected {raw_enc_col} or {fallback_enc_col}"
            )

        enc = pd.to_numeric(df[enc_col], errors="coerce")
        signed_pot = pd.to_numeric(df[pot_col], errors="coerce") * sign
        slope, intercept = _fit_linear_map(signed_pot, enc)
        mapped_pot = slope * signed_pot + intercept
        residual = mapped_pot - enc

        out[f"MAPPED_POT_{i}"] = mapped_pot.to_numpy(dtype=float)
        out[f"ENCODER_POS_{i}"] = enc.to_numpy(dtype=float)
        out[f"JOINT_{i}_RESIDUAL"] = residual.to_numpy(dtype=float)

    if pot_filter:
        fs = float(pot_original_freq) if pot_original_freq else _infer_sampling_rate(out["TIMESTAMP"])
        taps = _design_fir_filter(
            filter_type=pot_filter_type,
            fs=fs,
            cutoff_hz=float(pot_filter_cutoff_hz),
            order=int(pot_filter_order),
        )
        out = _apply_filter_to_extracted_df(out, taps)
        print(
            f"Applied POT filter: type={pot_filter_type}, cutoff={pot_filter_cutoff_hz:g} Hz, "
            f"order={pot_filter_order}, fs={fs:.6g} Hz"
        )

    if pot_downsample:
        if pot_downsample_freq is None:
            raise ValueError("--pot-downsample-freq is required when --pot-downsample is set.")
        fs_in = float(pot_original_freq) if pot_original_freq else _infer_sampling_rate(out["TIMESTAMP"])
        out = _downsample_df(
            out,
            original_freq=fs_in,
            target_freq=float(pot_downsample_freq),
            use_moving_average=pot_downsample_moving_average,
        )
        print(
            f"Applied POT downsample: {fs_in:.6g} Hz -> {pot_downsample_freq:g} Hz, "
            f"moving_average={pot_downsample_moving_average}"
        )

    out.to_csv(output_path, index=False)
    print(f"Saved {output_path}")

    if plot:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        tt = out["TIMESTAMP"].to_numpy(dtype=float)
        for i in range(3):
            ax = axes[i]
            ax.plot(tt, out[f"MAPPED_POT_{i + 1}"], label=f"MAPPED_POT_{i + 1}", linewidth=0.9)
            ax.plot(tt, out[f"ENCODER_POS_{i + 1}"], label=f"ENCODER_POS_{i + 1}", linewidth=0.9)
            ax.plot(tt, out[f"JOINT_{i + 1}_RESIDUAL"], label=f"JOINT_{i + 1}_RESIDUAL", linewidth=0.9)
            ax.set_title(f"Joint {i + 1}: mapped pot, encoder, residual")
            ax.grid(alpha=0.25)
            ax.legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("Timestamp")
        fig.tight_layout()
        out_png = output_path.with_name(f"{output_path.stem}_plot.png")
        fig.savefig(out_png, dpi=200)
        print(f"Saved {out_png}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract mapped POT, encoder position, and residual columns from a POT/encoder CSV."
        )
    )
    parser.add_argument("input_csv", type=str, help="Path to input CSV")
    parser.add_argument("output_csv", type=str, help="Path to extracted output CSV")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Optionally save a PNG showing mapped POT, encoder, and residual traces.",
    )
    parser.add_argument("--pot-filter", action="store_true", help="Apply FIR low-pass filter to extracted columns.")
    parser.add_argument("--pot-filter-cutoff-hz", type=float, default=30.0, help="POT filter cutoff in Hz.")
    parser.add_argument("--pot-filter-order", type=int, default=30, help="POT FIR filter order (numtaps = order + 1).")
    parser.add_argument(
        "--pot-filter-type",
        type=str,
        default="kaiser",
        choices=["kaiser", "hamming", "chebyshev"],
        help="POT FIR window type.",
    )
    parser.add_argument("--pot-downsample", action="store_true", help="Downsample extracted CSV.")
    parser.add_argument("--pot-downsample-freq", type=float, default=None, help="Target downsample frequency in Hz.")
    parser.add_argument(
        "--pot-original-freq",
        type=float,
        default=None,
        help="Original sampling frequency in Hz. If omitted, inferred from TIMESTAMP.",
    )
    parser.add_argument(
        "--pot-downsample-moving-average",
        action="store_true",
        help="Use moving-average window during downsampling.",
    )
    parser.add_argument(
        "--no-pot-downsample-moving-average",
        action="store_true",
        help="Disable moving-average window during downsampling.",
    )
    args = parser.parse_args()
    use_ma = True
    if args.pot_downsample_moving_average:
        use_ma = True
    if args.no_pot_downsample_moving_average:
        use_ma = False
    extract_encoder_info(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        plot=args.plot,
        pot_filter=args.pot_filter,
        pot_filter_cutoff_hz=args.pot_filter_cutoff_hz,
        pot_filter_order=args.pot_filter_order,
        pot_filter_type=args.pot_filter_type,
        pot_downsample=args.pot_downsample,
        pot_downsample_freq=args.pot_downsample_freq,
        pot_original_freq=args.pot_original_freq,
        pot_downsample_moving_average=use_ma,
    )


if __name__ == "__main__":
    main()
