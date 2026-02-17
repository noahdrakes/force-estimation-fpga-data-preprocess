#!/usr/bin/env python3
import os
import time
import argparse
import numpy as np
from pathlib import Path
from scipy import interpolate
from rosbags.highlevel import AnyReader


class Rosbag2Parser:
    def __init__(self, args):
        self.args = args
        for k, v in args.__dict__.items():
            setattr(self, k, v)

    def interp(self, time, mat):
        """Interpolate matrix columns to a given time vector."""
        new_mat = np.zeros((len(time), mat.shape[1]))
        new_mat[:, 0] = time
        for i in range(1, mat.shape[1]):
            f = interpolate.interp1d(
                mat[:, 0], mat[:, i], bounds_error=False, fill_value="extrapolate"
            )
            new_mat[:, i] = f(time)
        return new_mat

    def single_bag_to_csv(self, bag_path: Path):
        print(f"\n📦 Processing bag: {bag_path}")
        folder = Path(self.output)
        (folder / "joints").mkdir(parents=True, exist_ok=True)
        (folder / "jacobian").mkdir(parents=True, exist_ok=True)
        (folder / "sensor").mkdir(parents=True, exist_ok=True)
        (folder / "jaw").mkdir(parents=True, exist_ok=True)

        joint_timestamps, joint_position, joint_velocity, joint_effort = [], [], [], []
        jacobian_timestamps, jacobian_data = [], []
        force_timestamps, force_data = [], []
        jaw_timestamps, jaw_data = [], []

        with AnyReader([bag_path]) as reader:
            print(f"Opened bag with {len(reader.connections)} topics:")
            for c in reader.connections:
                print(" -", c.topic)

            for connection, timestamp, rawdata in reader.messages():
                msg = reader.deserialize(rawdata, connection.msgtype)
                t = timestamp / 1e9  # convert ns → s
                topic = connection.topic

                # --- Joint states ---
                if topic in ("PSM1/measured_js", "/PSM1/measured_js"):
                    joint_timestamps.append(t)
                    joint_position.append(list(msg.position))
                    joint_velocity.append(list(msg.velocity))
                    joint_effort.append(list(msg.effort))

                # --- Jacobian ---
                elif topic in ("PSM1/spatial/jacobian", "/PSM1/spatial/jacobian"):
                    jacobian_timestamps.append(t)
                    jacobian_data.append(list(msg.data))

                # --- Jaw joint state ---
                elif topic in ("PSM1/jaw/measured_js", "/PSM1/jaw/measured_js"):
                    jaw_timestamps.append(t)
                    jaw_data.append([msg.position, msg.velocity, msg.effort])

                # --- Force sensors ---
                elif topic in (
                    "/measured_cf",
                    "/PSM1/body/measured_cf",
                    "/PSM1/spatial/measured_cf",
                ):
                    force_timestamps.append(t)
                    f = msg.wrench.force
                    tau = msg.wrench.torque
                    force_data.append([f.x, f.y, f.z, tau.x, tau.y, tau.z])

        # ✅ check after all messages
        if not joint_timestamps:
            print("⚠️  No joint data found — skipping.")
            return

        # --- Normalize timestamps ---
        start_time = joint_timestamps[0]
        joint_timestamps = np.array(joint_timestamps) - start_time
        jacobian_timestamps = (
            np.array(jacobian_timestamps) - start_time if jacobian_timestamps else []
        )
        force_timestamps = (
            np.array(force_timestamps) - start_time if force_timestamps else []
        )
        jaw_timestamps = (
            np.array(jaw_timestamps) - start_time if jaw_timestamps else []
        )

        # --- Write CSVs ---
        joints = np.column_stack(
            (joint_timestamps, joint_position, joint_velocity, joint_effort)
        )
        np.savetxt(
            folder / "joints" / f"{self.prefix}{self.index}.csv", joints, delimiter=","
        )

        if len(jacobian_data) > 0:
            jacobian = np.column_stack((jacobian_timestamps, jacobian_data))
            np.savetxt(
                folder / "jacobian" / f"{self.prefix}{self.index}.csv",
                jacobian,
                delimiter=",",
            )

        if len(force_data) > 0:
            force = np.column_stack((force_timestamps, force_data))
            np.savetxt(
                folder / "sensor" / f"{self.prefix}{self.index}.csv",
                force,
                delimiter=",",
            )

        if len(jaw_data) > 0:
            jaw = np.column_stack((jaw_timestamps, np.squeeze(jaw_data)))
            np.savetxt(
                folder / "jaw" / f"{self.prefix}{self.index}.csv", jaw, delimiter=","
            )

        print(f"✅ Wrote out {self.prefix}{self.index}.csv")
        self.index += 1

    def parse_all(self):
        bag_path = Path(self.folder)
        if (bag_path / "metadata.yaml").exists():
            # Single ROS2 bag folder
            self.single_bag_to_csv(bag_path)
        else:
            # Search for nested bag folders
            for subdir in sorted(bag_path.iterdir()):
                if (subdir / "metadata.yaml").exists():
                    self.single_bag_to_csv(subdir)


def main():
    parser = argparse.ArgumentParser(
        description="Convert ROS2 bag (mcap/db3) to CSV folders."
    )
    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        required=True,
        help="Path to ROS2 bag folder (contains metadata.yaml)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./parsed_data/",
        help="Directory to save CSVs",
    )
    parser.add_argument(
        "--prefix", type=str, default="ros2_", help="Prefix for output CSV filenames"
    )
    parser.add_argument(
        "--index", type=int, default=0, help="Starting file index for naming"
    )
    args = parser.parse_args()

    start = time.time()
    Rosbag2Parser(args).parse_all()
    print(f"\n🏁 Parsing complete in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()