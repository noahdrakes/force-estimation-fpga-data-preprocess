import pandas as pd
import numpy as np
import argparse
import cisstRobotPython

def compute_flattened_jacobian(input_csv, output_csv, robot_file):
    # Load the robot model
    r = cisstRobotPython.robManipulator()
    if r.LoadRobot(robot_file) != 0:
        raise RuntimeError(f"Failed to load robot file: {robot_file}")

    # Load joint configurations from CSV
    df = pd.read_csv(input_csv, header=None)
    joint_configs = df.iloc[:, 1:7].values  # Extract joint positions (columns 1-6)

    # Prepare output data
    flattened_jacobians = []

    for jp in joint_configs:
        # Compute spatial Jacobian
        J_spatial = np.zeros((6, 6), dtype=np.double)
        r.JacobianSpatial(jp, J_spatial)

        # Flatten the Jacobian and append to output
        flattened_jacobians.append(J_spatial.flatten())

    # Save flattened Jacobians to CSV
    flattened_jacobians = np.array(flattened_jacobians)
    pd.DataFrame(flattened_jacobians).to_csv(output_csv, index=False, header=False)
    print(f"Flattened Jacobians saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute flattened Jacobians for joint configurations")
    parser.add_argument("input_csv", type=str, help="Path to interpolated_all_joints.csv")
    parser.add_argument("output_csv", type=str, help="Path to save flattened Jacobians")
    parser.add_argument("robot_file", type=str, help="Path to robot file (.rob)")
    args = parser.parse_args()

    compute_flattened_jacobian(args.input_csv, args.output_csv, args.robot_file)