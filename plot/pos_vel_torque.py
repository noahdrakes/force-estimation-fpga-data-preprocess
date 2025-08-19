import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

# Expected column order for interpolated_all_joints.csv
COLUMN_NAMES: List[str] = (
    ['TIMESTAMP'] +
    [f'POSITION_FEEDBACK_{i}' for i in range(1, 7)] +
    [f'VELOCITY_FEEDBACK_{i}' for i in range(1, 7)] +
    [f'TORQUE_FEEDBACK_{i}' for i in range(1, 7)]
)


def read_interpolated_csv(csv_path: Path) -> pd.DataFrame:
    """Read interpolated_all_joints.csv style file saved without headers."""
    return pd.read_csv(csv_path, header=None, names=COLUMN_NAMES)


def plot_position_velocity_torque(df: pd.DataFrame, out_png: Path = None, joint: int = 1):
    """Plot position, velocity, torque for a given joint and save or show the figure."""
    ts = df['TIMESTAMP']
    pos_col = f'POSITION_FEEDBACK_{joint}'
    vel_col = f'VELOCITY_FEEDBACK_{joint}'
    tor_col = f'TORQUE_FEEDBACK_{joint}'

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    ax1.plot(ts, df[pos_col])
    ax1.set_ylabel("Position")
    ax1.set_title(f"Joint {joint} Position, Velocity, Torque")

    ax2.plot(ts, df[vel_col])
    ax2.set_ylabel("Velocity")

    ax3.plot(ts, df[tor_col])
    ax3.set_ylabel("Torque")
    ax3.set_xlabel("Timestamp")


    plt.tight_layout()
    # plt.show()
    if out_png:
        plt.savefig(out_png, dpi=300)
        print(f"Saved plot to {out_png}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot position, velocity, torque for one joint from interpolated CSV.")
    parser.add_argument("--csv", required=True, help="Path to interpolated_all_joints.csv file.")
    parser.add_argument("--joint", type=int, default=1, help="Joint index (1-6) to plot. Default=1")
    parser.add_argument("--out", type=str, default=None, help="Output PNG file name. If not provided, show plot interactively.")
    args = parser.parse_args()

    df = read_interpolated_csv(Path(args.csv))
    out_path = Path(args.out) if args.out else None
    plot_position_velocity_torque(df, out_path, joint=args.joint)


if __name__ == "__main__":
    main()