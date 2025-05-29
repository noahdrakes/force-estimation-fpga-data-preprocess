import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description='Plot force estimation data (encoder pos/vel and motor state) from two csv file (one of the robot doing a motion in freespace and one in which the robot is performing the same motion but hitting an object).')
parser.add_argument('fileA', type=str, help='Path to the first CSV file (moving in free space))')

args = parser.parse_args()

# Load data from the CSV files
df = pd.read_csv(args.fileA)

# Only keep the first 6 columns for each feedback type
position_cols = [col for col in df.columns if 'POSITION_FEEDBACK' in col][:6]
velocity_cols = [col for col in df.columns if 'VELOCITY_FEEDBACK' in col][:6]
torque_cols = [col for col in df.columns if 'TORQUE_FEEDBACK' in col][:6]

ordered_cols = ['TIMESTAMP'] + position_cols + velocity_cols + torque_cols

# Do not drop the first row unless you know it's not data
df = df.drop(df.index[0])

df_ordered = df[ordered_cols]

df_ordered.to_csv('your_data_reordered.csv', index=False)

with open("your_data_reordered.csv",'r') as f:
    with open("interpolated_all_joints.csv",'w') as f1:
        next(f) # skip header line
        for line in f:
            f1.write(line)

os.remove("your_data_reordered.csv")

print(df_ordered.head())