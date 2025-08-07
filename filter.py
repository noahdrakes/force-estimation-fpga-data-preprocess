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

def apply_filter_to_torque_feedback(csv_path, output_path, fir_coeffs, fs_in=None, fs_out=None, preserve_rate=True):
    """
    Apply zero-phase FIR filter to torque feedback columns in a CSV file.
    
    Parameters:
        csv_path (str): Input CSV file path
        output_path (str): Output CSV file path
        fir_coeffs (array): FIR filter coefficients
        fs_in (float, optional): Original sampling rate (required if downsampling)
        fs_out (float, optional): Desired sampling rate (if downsampling)
        preserve_rate (bool): If False and fs_out is provided, will downsample
    """
    df = pd.read_csv(csv_path, header=None)
    
    torque_cols = list(range(13, 19))
    df.iloc[:, torque_cols] = filtfilt(fir_coeffs, [1.0], df.iloc[:, torque_cols], axis=0)
    
    if not preserve_rate and fs_in and fs_out:
        factor = int(fs_in // fs_out)
        df = df.iloc[::factor].reset_index(drop=True)

    df.to_csv(output_path, index=False, header=False)
    print(f"Filtered {'and downsampled ' if not preserve_rate and fs_out else ''}and saved to {output_path}")