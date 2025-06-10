import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Preprocess force estimation data.')
parser.add_argument('fileA', type=str, help='Path to the first CSV file (moving in free space))')
args = parser.parse_args()

# Load data with header row
df = pd.read_csv(args.fileA)

# Select columns by name (first 6 for each, only measured torque)
ordered_cols = (
    ['TIMESTAMP'] +
    [f'POSITION_FEEDBACK_{i}' for i in range(1, 7)] +
    [f'VELOCITY_FEEDBACK_{i}' for i in range(1, 7)] +
    [f'TORQUE_FEEDBACK_{i}' for i in range(1, 7)]
)

df_ordered = df[ordered_cols]
df_ordered.to_csv('interpolated_all_joints.csv', index=False, header=False)