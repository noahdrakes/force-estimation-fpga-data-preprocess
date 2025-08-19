

import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_joint_data(file_path: str, joint_idx: int):
    """
    Plot joint position, velocity, and torque for a specified joint index using subplots.
    
    Args:
        file_path: Path to the CSV file. 
                   Columns = timestamp | 6 joint pos | 6 joint vel | 6 joint torques
        joint_idx: Joint index to plot (1–6).
    """
    # Load data
    df = pd.read_csv(file_path, header=None)
    
    timestamps = df.iloc[:, 0].values
    positions = df.iloc[:, 1:7].values
    velocities = df.iloc[:, 7:13].values
    torques   = df.iloc[:, 13:19].values

    j = joint_idx - 1  # convert to 0-based index

    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    axs[0].plot(timestamps, positions[:, j])
    axs[0].set_ylabel("Position")
    axs[0].set_title(f"Joint {joint_idx}")

    axs[1].plot(timestamps, velocities[:, j])
    axs[1].set_ylabel("Velocity")

    axs[2].plot(timestamps, torques[:, j])
    axs[2].set_ylabel("Torque")
    axs[2].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot joint position, velocity, and torque for a specified joint index.")
    parser.add_argument("file_path", type=str, help="Path to the CSV file.")
    parser.add_argument("joint_idx", type=int, help="Joint index to plot (1-6).")
    args = parser.parse_args()

    plot_joint_data(args.file_path, args.joint_idx)