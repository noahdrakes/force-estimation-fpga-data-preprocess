import pandas as pd

def truncate_dataframe(df, seconds_to_trim, frequency):
    total_rows = len(df)
    print("total rows: ", total_rows)
    rows_to_trim = int(seconds_to_trim * frequency)
    total_rows_to_remove = 2 * rows_to_trim
    if total_rows_to_remove >= total_rows:
        raise ValueError("Cannot remove more rows than available in the dataframe")
    df_truncated = df.iloc[rows_to_trim:total_rows - rows_to_trim]
    return df_truncated

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=str)
    parser.add_argument("output_csv", type=str)
    parser.add_argument("--seconds_to_trim", type=float, required=True)
    parser.add_argument("--frequency", type=float, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv, header=None)
    df_truncated = truncate_dataframe(df, args.seconds_to_trim, args.frequency)
    df_truncated.to_csv(args.output_csv, index=False, header=False)
    print(f"Saved truncated CSV to {args.output_csv}, removed {args.seconds_to_trim} seconds from beginning and end")
