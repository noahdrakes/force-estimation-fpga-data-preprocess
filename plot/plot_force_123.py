import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def read_force_csv(csv_path: Path, has_header: bool = True) -> pd.DataFrame:
    """Read CSV and return dataframe with FORCE_1/2/3 columns present."""
    if has_header:
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(csv_path, header=None)
    return df


def plot_force_123(df: pd.DataFrame, out_png: Path = None):
    for col in ("FORCE_1", "FORCE_2", "FORCE_3"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    ts = df["TIMESTAMP"] if "TIMESTAMP" in df.columns else None

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    labels = ["FORCE_1", "FORCE_2", "FORCE_3"]
    x_label = "Timestamp" if ts is not None else "Sample Index"

    for ax, col in zip(axes, labels):
        if ts is not None:
            ax.plot(ts, df[col])
        else:
            ax.plot(df[col])
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(x_label)
    axes[0].set_title("Forces by Axis")
    plt.tight_layout()

    if out_png:
        plt.savefig(out_png, dpi=300)
        print(f"Saved plot to {out_png}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot FORCE_1/2/3 from a CSV.")
    parser.add_argument("--csv", required=True, help="Path to CSV file.")
    parser.add_argument("--out", type=str, default=None, help="Output PNG file name. If not provided, show plot interactively.")
    parser.add_argument("--no-header", action="store_true", help="CSV has no header row.")
    args = parser.parse_args()

    df = read_force_csv(Path(args.csv), has_header=not args.no_header)
    out_path = Path(args.out) if args.out else None
    plot_force_123(df, out_path)


if __name__ == "__main__":
    main()
