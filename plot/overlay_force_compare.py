import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def read_original(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def read_filtered_no_header(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=None)
    n_cols = df.shape[1]
    columns = ["TIMESTAMP"] + [f"COL_{i}" for i in range(1, n_cols)]
    df.columns = columns
    return df


def plot_overlay(
    original: pd.DataFrame,
    filtered: pd.DataFrame,
    out_png: Path = None,
    orig_cols=("FORCE_1", "FORCE_2", "FORCE_3"),
    filt_cols=("COL_1", "COL_2", "COL_3"),
):
    for col in orig_cols:
        if col not in original.columns:
            raise ValueError(f"Missing original column: {col}")
    for col in filt_cols:
        if col not in filtered.columns:
            raise ValueError(f"Missing filtered column: {col}")

    ts_orig = original["TIMESTAMP"] if "TIMESTAMP" in original.columns else None
    ts_filt = filtered["TIMESTAMP"] if "TIMESTAMP" in filtered.columns else None

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axis_labels = ["X", "Y", "Z"]

    for i, ax in enumerate(axes):
        oc = orig_cols[i]
        fc = filt_cols[i]
        if ts_orig is not None:
            ax.plot(ts_orig, original[oc], label="Original", linewidth=0.8)
        else:
            ax.plot(original[oc], label="Original", linewidth=0.8)

        if ts_filt is not None:
            ax.plot(ts_filt, filtered[fc], label="Filtered", linewidth=0.8)
        else:
            ax.plot(filtered[fc], label="Filtered", linewidth=0.8)

        ax.set_ylabel(f"Force {axis_labels[i]}")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Timestamp")
    axes[0].set_title("Overlay: Original vs Filtered Forces")
    axes[0].legend()
    plt.tight_layout()

    if out_png:
        plt.savefig(out_png, dpi=300)
        print(f"Saved plot to {out_png}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Overlay original FORCE_1/2/3 with filtered no-header forces (COL_1/2/3)."
    )
    parser.add_argument("--original", required=True, help="Path to original CSV with headers.")
    parser.add_argument("--filtered", required=True, help="Path to filtered CSV without headers.")
    parser.add_argument("--out", type=str, default=None, help="Output PNG file. If not provided, show plot.")
    args = parser.parse_args()

    original = read_original(Path(args.original))
    filtered = read_filtered_no_header(Path(args.filtered))
    out_path = Path(args.out) if args.out else None
    plot_overlay(original, filtered, out_path)


if __name__ == "__main__":
    main()
