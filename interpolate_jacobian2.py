import pandas as pd
import numpy as np
import argparse
import cisstRobotPython

def compute_flattened_jacobian(input_csv, output_csv, robot_file):
    # Load the robot model
    r = cisstRobotPython.robManipulator()
    if r.LoadRobot(robot_file) != 0:
        raise RuntimeError(f"Failed to load robot file: {robot_file}")

    # Load joint configurations and timestamps
    df = pd.read_csv(input_csv)
    timestamps = df["TIMESTAMP"].to_numpy()
    joint_configs = df[
        [
            "POSITION_FEEDBACK_1",
            "POSITION_FEEDBACK_2",
            "POSITION_FEEDBACK_3",
            "POSITION_FEEDBACK_4",
            "POSITION_FEEDBACK_5",
            "POSITION_FEEDBACK_6",
        ]
    ].to_numpy()

    rows = []
    i = 0
    for ts, jp in zip(timestamps, joint_configs):
        J = np.zeros((6, 6), dtype=np.float64)
        r.JacobianSpatial(jp, J)  # fills J in-place

        if i == 0:
            print("FIRST JACOBIAN VALUE")
            print(J) 
        # exit()

        # >>> KEY CHANGE: flatten in column-major (Fortran) order <<<
        row = np.concatenate(([ts], J.flatten(order="C")))
        rows.append(row)
        i+=1

    arr = np.asarray(rows)

    # Optional: write a header (timestamp + J11..J66 in column-major order)
    # header = ["TIMESTAMP"] + [f"J{r}{c}" for c in range(1,7) for r in range(1,7)]
    pd.DataFrame(arr).to_csv(output_csv, index=False, header=False)
    print(f"Flattened Jacobians (column-major) written to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute flattened Jacobians for joint configurations")
    parser.add_argument("input_csv", type=str, help="Path to unit converted .csv . should be in the format capture_unitConvert.csv")
    parser.add_argument("output_csv", type=str, help="Path to save flattened Jacobians")
    parser.add_argument("robot_file", type=str, help="Path to robot file (.rob)")
    args = parser.parse_args()

    compute_flattened_jacobian(args.input_csv, args.output_csv, args.robot_file)