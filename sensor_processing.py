import pandas as pd

def save_sensor_data(df, sensor_cols, output_path):
    sensor_df = df[sensor_cols]

    sensor_df.to_csv(output_path, index=False, header=False)
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=str)
    parser.add_argument("output_csv", type=str)

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    sensor_columns = ["TIMESTAMP","FORCE_1", "FORCE_2", "FORCE_3", "TORQUE_1", "TORQUE_2", "TORQUE_3"]
    save_sensor_data(df, sensor_columns, args.output_csv)

    print(f"Saved SENSOR DATA CSV to {args.output_csv}")

