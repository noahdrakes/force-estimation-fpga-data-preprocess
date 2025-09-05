import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import argparse

def interpolate_dataframe_to_sample_rate(df, target_sample_rate):
    """
    Interpolate the DataFrame to match the given target sample rate.

    Parameters:
    - df: pd.DataFrame, with a time column in seconds (float64) as the first column.
    - target_sample_rate: float, desired sample rate in Hz.

    Returns:
    - interpolated_df: pd.DataFrame with interpolated data at the new sample rate.
    """
    df.columns = ['time'] + [f'col_{i}' for i in range(1, df.shape[1])]
    time = df['time'].values.astype(np.float64)
    shifted_time = time - time[0]
    print(shifted_time)

    duration = shifted_time[-1]
    n_new_samples = int(np.floor(duration * target_sample_rate)) + 1
    new_time = np.linspace(0, duration, n_new_samples)

    interp_func = interp1d(shifted_time, df.drop(columns='time').values.T, kind='linear', fill_value="extrapolate")
    interpolated_values = interp_func(new_time).T

    interpolated_df = pd.DataFrame(np.column_stack([new_time, interpolated_values]), columns=df.columns)
    return interpolated_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpolate CSV timestamps to a target sample rate.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file.")
    parser.add_argument("--sample_rate", type=float, required=True, help="Target sample rate in Hz.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path, header=None)
    interpolated_df = interpolate_dataframe_to_sample_rate(df, args.sample_rate)
    interpolated_df.to_csv(args.csv_path, index=False, header=False)
    print(f"Interpolated and saved to {args.csv_path} at {args.sample_rate} Hz")
