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

def apply_filter_to_torque_feedback_df(df, fir_coeffs):
    """
    Apply zero-phase FIR filter to torque feedback columns in a DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        fir_coeffs (array): FIR filter coefficients

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    torque_cols = list(range(13, 19))
    df.iloc[:, torque_cols] = filtfilt(fir_coeffs, [1.0], df.iloc[:, torque_cols], axis=0)
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
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv, header=None)
    fir_coeffs = design_fir_filter(args.filter_type, args.fs, args.fC, args.order)
    df_filtered = apply_filter_to_torque_feedback_df(df, fir_coeffs)
    df_filtered.to_csv(args.output_csv, index=False, header=False)
    print(f"Filtered and saved to {args.output_csv}")