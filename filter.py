import numpy as np
from scipy.signal import firwin, filtfilt
from scipy.signal.windows import kaiser, hamming, chebwin
import pandas as pd

def design_fir_filter(filter_type: str, fs: float, fC: float, order: int):
    fC_norm = fC / (fs / 2)  # Normalize cutoff frequency

    if filter_type == 'kaiser':
        beta = 3.5
        return firwin(order + 1, fC_norm, window=('kaiser', beta))
    elif filter_type == 'chebyshev':
        atten = 40
        return firwin(order + 1, fC_norm, window=('chebwin', atten))
    elif filter_type == 'hamming':
        return firwin(order + 1, fC_norm, window='hamming')
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

def apply_filter_to_torque_feedback_df(df, fir_coeffs, filter_velocity=False, filter_position=False):
    """
    Apply zero-phase FIR filter to torque feedback columns in a DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        fir_coeffs (array): FIR filter coefficients
        filter_velocity (bool): If True, also apply filtering to velocity columns (7 through 12)
        filter_position (bool): If True, also apply filtering to position columns (1 through 6)

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    torque_cols = list(range(13, 19))
    df.iloc[:, torque_cols] = filtfilt(fir_coeffs, [1.0], df.iloc[:, torque_cols], axis=0)
    if filter_velocity:
        velocity_cols = list(range(7, 13))
        df.iloc[:, velocity_cols] = filtfilt(fir_coeffs, [1.0], df.iloc[:, velocity_cols], axis=0)
    if filter_position:
        position_cols = list(range(1, 7))
        df.iloc[:, position_cols] = filtfilt(fir_coeffs, [1.0], df.iloc[:, position_cols], axis=0)
    return df

def apply_filter_to_fs_df(df, fir_coeffs):
    """
    Apply zero-phase FIR filter to torque feedback columns in a DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        fir_coeffs (array): FIR filter coefficients
        filter_velocity (bool): If True, also apply filtering to velocity columns (7 through 12)

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    force_torque_cols = list(range(1, 7))
    df.iloc[:, force_torque_cols] = filtfilt(fir_coeffs, [1.0], df.iloc[:, force_torque_cols], axis=0)
    return df



# If run as a script, provide CLI for file I/O
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=str)
    parser.add_argument("output_csv", type=str)
    parser.add_argument("--filter_type", type=str, default="kaiser")
    parser.add_argument("--fs", type=float, required=True)
    parser.add_argument("--fC", type=float, required=True)
    parser.add_argument("--order", type=int, default=30)
    parser.add_argument("--filter_velocity", action="store_true", help="Also filter velocity columns (7–12)")
    parser.add_argument("--filter_position", action="store_true", help="Also filter position columns (1–6)")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv, header=None)
    fir_coeffs = design_fir_filter(args.filter_type, args.fs, args.fC, args.order)
    df_filtered = apply_filter_to_torque_feedback_df(
        df,
        fir_coeffs,
        filter_velocity=args.filter_velocity,
        filter_position=args.filter_position,
    )
    df_filtered.to_csv(args.output_csv, index=False, header=False)
    print(f"Filtered and saved to {args.output_csv}")
