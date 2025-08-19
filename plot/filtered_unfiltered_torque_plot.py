

import argparse
from pathlib import Path
from typing import List
import pandas as pd

try:
    import matplotlib.pyplot as plt
    _HAVE_PLT = True
except Exception:
    _HAVE_PLT = False


# Expected column order for interpolated_all_joints.csv
COLUMN_NAMES: List[str] = (
    ['TIMESTAMP'] +
    [f'POSITION_FEEDBACK_{i}' for i in range(1, 7)] +
    [f'VELOCITY_FEEDBACK_{i}' for i in range(1, 7)] +
    [f'TORQUE_FEEDBACK_{i}' for i in range(1, 7)]
)

TORQUE_COLS_1_TO_3 = [f'TORQUE_FEEDBACK_{i}' for i in range(1, 4)]


def read_interpolated_csv(csv_path: Path) -> pd.DataFrame:
    """Read interpolated_all_joints.csv style file saved without headers."""
    return pd.read_csv(csv_path, header=None, names=COLUMN_NAMES)


def select_measured_torque_1_to_3(df: pd.DataFrame) -> pd.DataFrame:
    """Return TIMESTAMP plus measured torque columns 1..3."""
    return df[['TIMESTAMP'] + TORQUE_COLS_1_TO_3].copy()


def join_on_timestamp(filtered_df: pd.DataFrame, unfiltered_df: pd.DataFrame) -> pd.DataFrame:
    """Join filtered and unfiltered torque columns 1..3 on TIMESTAMP."""
    f = filtered_df.set_index('TIMESTAMP')[TORQUE_COLS_1_TO_3].add_prefix('filtered_')
    u = unfiltered_df.set_index('TIMESTAMP')[TORQUE_COLS_1_TO_3].add_prefix('unfiltered_')
    joined = f.join(u, how='inner')
    joined.index.name = 'TIMESTAMP'
    return joined


def print_measured_torque_1_to_3(filtered_csv_path: str, unfiltered_csv_path: str, rows: int = 10) -> pd.DataFrame:
    """Load two CSVs, extract measured torque J1..J3, join on TIMESTAMP, print and return the DataFrame."""
    f_df = read_interpolated_csv(Path(filtered_csv_path))
    u_df = read_interpolated_csv(Path(unfiltered_csv_path))
    joined = join_on_timestamp(select_measured_torque_1_to_3(f_df), select_measured_torque_1_to_3(u_df))
    print(joined.head(rows).to_string())
    return joined


def maybe_plot(joined: pd.DataFrame, out_png: Path = None):
    """Optionally plot filtered vs unfiltered torque for joints 1..3."""
    if not _HAVE_PLT:
        print("matplotlib not available; skipping plot.")
        return

    df = joined.reset_index()
    ts = df['TIMESTAMP']

    for j in range(1, 4):
        col_f = f'filtered_TORQUE_FEEDBACK_{j}'
        col_u = f'unfiltered_TORQUE_FEEDBACK_{j}'
        if col_f not in df.columns or col_u not in df.columns:
            continue
        plt.figure()
        plt.plot(ts, df[col_f], label='filtered')
        plt.plot(ts, df[col_u], label='unfiltered')
        plt.xlabel('TIMESTAMP')
        plt.ylabel(f'Torque J{j}')
        plt.title(f'Filtered vs Unfiltered Torque (Joint {j})')
        plt.legend()
        plt.tight_layout()
        if out_png:
            stem = out_png.stem
            joint_png = out_png.with_name(f"{stem}_J{j}{out_png.suffix}")
            plt.savefig(joint_png, dpi=300)

    if out_png is None:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Print and plot measured torque J1..J3 for filtered vs unfiltered CSVs.")
    parser.add_argument("--filtered", required=True, help="Path to filtered interpolated CSV.")
    parser.add_argument("--unfiltered", required=True, help="Path to unfiltered interpolated CSV.")
    parser.add_argument("--rows", type=int, default=10, help="Number of rows to print from the joined view.")
    parser.add_argument("--plot", action="store_true", help="If set, plot filtered vs unfiltered torque (J1..J3).")
    parser.add_argument("--out", type=str, default="", help="Optional output PNG path base (e.g., /path/torque_compare.png).")
    args = parser.parse_args()

    joined = print_measured_torque_1_to_3(args.filtered, args.unfiltered, rows=args.rows)

    if args.plot:
        out_png = Path(args.out) if args.out else None
        maybe_plot(joined, out_png=out_png)


if __name__ == "__main__":
    main()