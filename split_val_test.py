import pandas as pd
import argparse

def split_val_test(input_csv, split_ratio):
    df = pd.read_csv(input_csv, header=None)
    split_idx = int(len(df) * split_ratio)
    val_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Shift test timestamps so first is 0 (assuming timestamp is column 0)
    test_df[0] = test_df[0] - test_df[0].iloc[0]

    val_df.to_csv("val.csv", index=False, header=False)
    test_df.to_csv("test.csv", index=False, header=False)
    print("Validation set saved to val.csv", len(val_df), "rows")
    print("Test set saved to test.csv", len(test_df), "rows")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split preprocessed CSV into validation and test sets (no shuffle, preserves order)")
    parser.add_argument("input_csv", type=str, help="Path to preprocessed CSV file")
    parser.add_argument("--split_ratio", type=float, default=0.5, help="Fraction of data for validation set (default: 0.5)")
    args = parser.parse_args()
    split_val_test(args.input_csv, args.split_ratio)