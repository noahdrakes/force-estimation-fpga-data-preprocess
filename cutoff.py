import pandas as pd

def truncate_csv(input_file, output_file, seconds_to_remove, frequency):
    df = pd.read_csv(input_file, header=None)
    total_rows = len(df)
    rows_to_remove = int(seconds_to_remove * frequency)
    if rows_to_remove >= total_rows:
        raise ValueError("Cannot remove more rows than available in the file")
    df_truncated = df.iloc[:total_rows - rows_to_remove]
    df_truncated.to_csv(output_file, index=False, header=False)
    print(f"Saved truncated CSV to {output_file}, removed last {seconds_to_remove} seconds")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=str)
    parser.add_argument("output_csv", type=str)
    parser.add_argument("--seconds_to_remove", type=float, required=True)
    parser.add_argument("--frequency", type=float, required=True)
    args = parser.parse_args()

    truncate_csv(args.input_csv, args.output_csv, args.seconds_to_remove, args.frequency)
