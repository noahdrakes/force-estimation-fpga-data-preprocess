import pandas as pd
import argparse

def preprocess_csv(input_csv_path: str, output_csv_path: str = 'interpolated_all_joints.csv') -> pd.DataFrame:
    # Load data with header row
    df = pd.read_csv(input_csv_path)

    # Select columns by name (first 6 for each, only measured torque)
    ordered_cols = (
        ['TIMESTAMP'] +
        [f'POSITION_FEEDBACK_{i}' for i in range(1, 7)] +
        [f'VELOCITY_FEEDBACK_{i}' for i in range(1, 7)] +
        [f'TORQUE_FEEDBACK_{i}' for i in range(1, 7)]
    )

    df_ordered = df[ordered_cols]
    df_ordered.to_csv(output_csv_path, index=False, header=False)
    return df_ordered

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess force estimation data.')
    parser.add_argument('fileA', type=str, help='Path to the first CSV file (moving in free space)')
    parser.add_argument("output_path", type=str, help="output_path")
    args = parser.parse_args()
    preprocess_csv(args.fileA, args.output_path)