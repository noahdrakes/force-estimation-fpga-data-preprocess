import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_force_csv(csv_path: Path, has_header: bool = True) -> pd.DataFrame:
    if has_header:
        return pd.read_csv(csv_path)

    df = pd.read_csv(csv_path, header=None)
    n_cols = df.shape[1]
    columns = ["TIMESTAMP"] + [f"COL_{i}" for i in range(1, n_cols)]
    df.columns = columns
    return df


def compute_fft(signal: np.ndarray, fs: float):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_vals = np.fft.rfft(signal)
    mag = np.abs(fft_vals) / n
    return freqs, mag


def infer_fs_from_timestamp(df: pd.DataFrame) -> float:
    if "TIMESTAMP" not in df.columns:
        raise ValueError("TIMESTAMP column required to infer sampling frequency.")
    ts = df["TIMESTAMP"].to_numpy()
    if len(ts) < 2:
        raise ValueError("Need at least two samples to infer sampling frequency.")
    dt = np.diff(ts)
    dt = dt[dt > 0]
    if len(dt) == 0:
        raise ValueError("Invalid TIMESTAMP values for inferring sampling frequency.")
    median_dt = float(np.median(dt))
    if median_dt <= 0:
        raise ValueError("Invalid TIMESTAMP spacing for inferring sampling frequency.")
    return 1.0 / median_dt


def plot_fft_columns(df: pd.DataFrame, columns: list[str], out_png: Path = None):
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    fs = infer_fs_from_timestamp(df)
    n_plots = len(columns)

    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 + 2 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        signal = df[col].to_numpy()
        freqs, mag = compute_fft(signal, fs)
        ax.plot(freqs, mag)
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frequency (Hz)")
    axes[0].set_title("FFT Magnitude")
    plt.tight_layout()

    if out_png:
        plt.savefig(out_png, dpi=300)
        print(f"Saved plot to {out_png}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="FFT of a specified column from a CSV.")
    parser.add_argument("--csv", required=True, help="Path to CSV file.")
    parser.add_argument(
        "--columns",
        default=None,
        help="Comma-separated column names to FFT (e.g., FORCE_1,FORCE_2).",
    )
    parser.add_argument(
        "--col-indices",
        default=None,
        help="Comma-separated 0-based column indices (excluding timestamp) for no-header CSVs, e.g. 1,2,3.",
    )
    parser.add_argument("--out", type=str, default=None, help="Output PNG file name. If not provided, show plot interactively.")
    parser.add_argument("--no-header", action="store_true", help="CSV has no header row.")
    args = parser.parse_args()

    df = read_force_csv(Path(args.csv), has_header=not args.no_header)
    out_path = Path(args.out) if args.out else None
    if args.columns:
        columns = [c.strip() for c in args.columns.split(",") if c.strip()]
    elif args.col_indices:
        indices = [i.strip() for i in args.col_indices.split(",") if i.strip()]
        try:
            indices_int = [int(i) for i in indices]
        except ValueError as exc:
            raise ValueError("col-indices must be integers, e.g. 1,2,3") from exc
        columns = []
        for i in indices_int:
            if i == 0:
                raise ValueError("Index 0 is TIMESTAMP. Use data columns only.")
            columns.append(f"COL_{i}")
    else:
        raise ValueError("No columns specified. Use --columns or --col-indices.")

    plot_fft_columns(df, columns, out_path)


if __name__ == "__main__":
    main()
