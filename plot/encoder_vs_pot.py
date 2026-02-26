#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot encoder vs pot signals from *_potEncoder.csv")
    parser.add_argument("csv", type=str, help="Path to *_potEncoder.csv")
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    df = pd.read_csv(csv_path)

    t = pd.to_numeric(df["TIMESTAMP"], errors="coerce")
    t = t - t.iloc[0]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    pairs = [("POT_3", "ENCODER_POS_1", "ORIGINAL_ENCODER_POS_1"),
             ("POT_4", "ENCODER_POS_2", "ORIGINAL_ENCODER_POS_2"),
             ("POT_5", "ENCODER_POS_3", "ORIGINAL_ENCODER_POS_3")]

    for i, (pot, enc, enc_raw) in enumerate(pairs, start=1):
        ax = axes[i - 1]
        ax.plot(t, df[pot], label=pot, linewidth=0.9)
        ax.plot(t, df[enc], label=f"{enc} (mapped)", linewidth=0.9)
        ax.plot(t, df[enc_raw], label=f"{enc_raw} (raw)", linewidth=0.9, alpha=0.7)
        ax.set_title(f"Joint {i}")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time from start")
    fig.tight_layout()

    out_png = csv_path.with_name(f"{csv_path.stem}_encoder_vs_pot.png")
    fig.savefig(out_png, dpi=200)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
