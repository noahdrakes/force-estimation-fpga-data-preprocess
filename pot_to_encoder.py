import argparse
import numpy as np
import pandas as pd


def _fit_linear_map(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    x_num = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_num = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(x_num) & np.isfinite(y_num)
    if np.count_nonzero(valid) < 2:
        return 1.0, 0.0

    x_valid = x_num[valid]
    y_valid = y_num[valid]

    x_mean = x_valid.mean()
    y_mean = y_valid.mean()
    denom = np.sum((x_valid - x_mean) ** 2)
    if denom <= np.finfo(float).eps:
        return 1.0, float(y_mean - x_mean)

    slope = float(np.sum((x_valid - x_mean) * (y_valid - y_mean)) / denom)
    intercept = float(y_mean - slope * x_mean)
    return slope, intercept


def _smoothed_velocity(
    timestamps: pd.Series,
    positions: pd.Series,
    smooth_span: int,
) -> np.ndarray:
    ts = pd.to_numeric(timestamps, errors="coerce").to_numpy(dtype=float)
    pos = pd.to_numeric(positions, errors="coerce").to_numpy(dtype=float)

    if len(pos) < 3:
        return np.zeros_like(pos)

    span = max(3, int(smooth_span))
    pos_smooth = pd.Series(pos).ewm(span=span, adjust=False).mean().to_numpy()

    dt = np.diff(ts)
    if np.all(dt > 0):
        vel = np.gradient(pos_smooth, ts, edge_order=1)
    else:
        positive_dt = dt[dt > 0]
        step = float(np.median(positive_dt)) if len(positive_dt) else 1.0
        vel = np.gradient(pos_smooth, step, edge_order=1)

    vel_smooth = pd.Series(vel).ewm(span=max(3, span // 2), adjust=False).mean().to_numpy()
    return vel_smooth


def replace_encoder_from_pots(
    input_csv: str,
    output_csv: str,
    update_velocity: bool = True,
    vel_smooth_span: int = 25,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    required_cols = [
        "TIMESTAMP",
        "POT_3",
        "POT_4",
        "POT_5",
        "ENCODER_POS_1",
        "ENCODER_POS_2",
        "ENCODER_POS_3",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep the raw encoder position channels in the output for reference.
    for i in range(1, 4):
        df[f"ORIGINAL_ENCODER_POS_{i}"] = df[f"ENCODER_POS_{i}"]

    pot_to_encoder_pos = [
        ("POT_3", "ENCODER_POS_1", 1.0),
        ("POT_4", "ENCODER_POS_2", 1.0),
        ("POT_5", "ENCODER_POS_3", -1.0),
    ]

    for pot_col, enc_pos_col, sign in pot_to_encoder_pos:
        signed_pot = df[pot_col] * sign
        slope, intercept = _fit_linear_map(signed_pot, df[enc_pos_col])
        df[enc_pos_col] = slope * signed_pot + intercept

    if update_velocity:
        vel_cols = [f"ENCODER_VEL_{i}" for i in range(1, 4)]
        missing_vel = [c for c in vel_cols if c not in df.columns]
        if missing_vel:
            raise ValueError(
                f"Velocity update requested, but missing velocity columns: {missing_vel}"
            )
        for i in range(1, 4):
            df[f"ENCODER_VEL_{i}"] = _smoothed_velocity(
                timestamps=df["TIMESTAMP"],
                positions=df[f"ENCODER_POS_{i}"],
                smooth_span=vel_smooth_span,
            )

    df.to_csv(output_csv, index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replace ENCODER_POS_1/2/3 using linearly mapped POT_3/4/inverted POT_5 before unit conversion."
    )
    parser.add_argument("input_csv", type=str, help="Path to raw CSV with headers")
    parser.add_argument("output_csv", type=str, help="Path to save transformed CSV")
    parser.add_argument(
        "--keep-original-vel",
        action="store_true",
        help="Do not recompute ENCODER_VEL_1/2/3 from the POT-derived positions.",
    )
    parser.add_argument(
        "--vel-smooth-span",
        type=int,
        default=25,
        help="Smoothing span used for POT-derived velocity calculation.",
    )
    args = parser.parse_args()

    replace_encoder_from_pots(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        update_velocity=not args.keep_original_vel,
        vel_smooth_span=args.vel_smooth_span,
    )
