import pandas as pd
import argparse

def downsample_dataframe(df, original_freq, target_freq, use_moving_average=False):
    # Compute window size
    window_size = int(original_freq / target_freq)
    if window_size < 1:
        raise ValueError("Target frequency must be lower than original frequency")

    # Separate timestamp column (assume it's the first column)
    timestamps = df.iloc[:, 0]
    data = df.iloc[:, 1:]

    if use_moving_average:
        n_windows = len(df) // window_size
        timestamps_downsampled = timestamps.iloc[:n_windows * window_size].groupby(
            timestamps.index[:n_windows * window_size] // window_size).mean().reset_index(drop=True)
        data_downsampled = data.iloc[:n_windows * window_size].groupby(
            data.index[:n_windows * window_size] // window_size).mean().reset_index(drop=True)
    else:
        timestamps_downsampled = timestamps.iloc[::window_size].reset_index(drop=True)
        data_downsampled = data.iloc[::window_size].reset_index(drop=True)

    # Combine timestamps and data
    df_downsampled = pd.concat([timestamps_downsampled, data_downsampled], axis=1)
    return df_downsampled


# MAIN
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample CSV with optional moving average")
    parser.add_argument("input_csv", type=str, help="Path to input CSV file")
    parser.add_argument("output_csv", type=str, help="Path to save downsampled CSV")
    parser.add_argument("--original_freq", type=float, required=True, help="Original frequency (Hz)")
    parser.add_argument("--target_freq", type=float, required=True, help="Target downsample frequency (Hz)")
    parser.add_argument("--use_moving_average", action='store_true', help="Use moving average for downsampling")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv, header=None)
    df_downsampled = downsample_dataframe(df, args.original_freq, args.target_freq, use_moving_average=args.use_moving_average)
    df_downsampled.to_csv(args.output_csv, index=False, header=False)
    print(f"Saved downsampled CSV to {args.output_csv}")