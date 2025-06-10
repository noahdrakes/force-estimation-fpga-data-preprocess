import pandas as pd
import argparse

# MOVING AVERAGE
def downsample_csv(input_file, output_file, original_freq, target_freq):
    df = pd.read_csv(input_file, header=None)

    # Compute window size
    window_size = int(original_freq / target_freq)
    if window_size < 1:
        raise ValueError("Target frequency must be lower than original frequency")

    # Separate timestamp column (assume it's the first column)
    timestamps = df.iloc[:, 0]
    data = df.iloc[:, 1:]

    # Apply moving average filter to all data columns
    data = data.rolling(window=window_size, center=True).mean()

    # Downsample
    timestamps_downsampled = timestamps.iloc[::window_size].reset_index(drop=True)
    data_downsampled = data.iloc[::window_size].reset_index(drop=True)

    # Combine timestamps and data
    df_downsampled = pd.concat([timestamps_downsampled, data_downsampled], axis=1)

    # Drop rows that are all NaN (except timestamps)
    df_downsampled = df_downsampled.dropna(how='all', subset=df_downsampled.columns[1:])

    # Save output without header and index
    df_downsampled.to_csv(output_file, index=False, header=False)
    print(f"Saved downsampled CSV to {output_file}")


# MAIN
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply moving average filter and downsample CSV")
    parser.add_argument("input_csv", type=str, help="Path to input CSV file")
    parser.add_argument("output_csv", type=str, help="Path to save filtered and downsampled CSV")
    parser.add_argument("--original_freq", type=float, required=True, help="Original frequency (Hz)")
    parser.add_argument("--target_freq", type=float, required=True, help="Target downsample frequency (Hz)")
    args = parser.parse_args()
    
    downsample_csv(args.input_csv, args.output_csv, args.original_freq, args.target_freq)
    print("done")