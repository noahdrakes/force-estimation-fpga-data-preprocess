#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
import argparse
import time
from scipy import interpolate

from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr


class Rosbag2Parser:
    def __init__(self, args):
        self.args = args
        for k, v in vars(args).items():
            setattr(self, k, v)

    def interp(self, time, mat):
        new_mat = np.zeros((len(time), mat.shape[1]))
        new_mat[:, 0] = time
        for i in range(mat.shape[1]):
            f = interpolate.interp1d(mat[:, 0], mat[:, i], fill_value="extrapolate")
            new_mat[:, i] = f(time)
        return new_mat

    def single_datapoint_processing(self, bag_folder):
        print(f"Processing {bag_folder}")

        reader = Reader(bag_folder)

        force_sensor, force_t = [], []
        joint_position, joint_velocity, joint_effort, joint_t = [], [], [], []
        jacobian, jacobian_t = [], []
        jaw, jaw_t = [], []

        with reader:
            for connection, timestamp, rawdata in reader.messages():
                topic = connection.topic
                msgtype = connection.msgtype
                msg = deserialize_cdr(rawdata, connection.msgtype)
                t_sec = timestamp * 1e-9  # convert ns → seconds

                if topic.endswith('measured_js') and 'jaw' not in topic:
                    joint_t.append(t_sec)
                    joint_position.append(list(msg.position))
                    joint_velocity.append(list(msg.velocity))
                    joint_effort.append(list(msg.effort))

                elif 'spatial/jacobian' in topic:
                    jacobian_t.append(t_sec)
                    jacobian.append(list(msg.data))

                elif 'jaw/measured_js' in topic:
                    jaw_t.append(t_sec)
                    jaw.append([msg.position, msg.velocity, msg.effort])

                elif topic.endswith('measured_cf'):
                    force_t.append(t_sec)
                    f = msg.wrench.force
                    tau = msg.wrench.torque
                    force_sensor.append([f.x, f.y, f.z, tau.x, tau.y, tau.z])

        print(f"Processed wrench: {len(force_t)} samples")
        print(f"Processed joints: {len(joint_t)} samples")
        print(f"Processed jaw: {len(jaw_t)} samples")
        print(f"Processed Jacobian: {len(jacobian_t)} samples")

        # Create output folders
        for sub in ['joints', 'jacobian', 'sensor', 'jaw']:
            (Path(self.output) / sub).mkdir(parents=True, exist_ok=True)

        start_time = joint_t[0] if joint_t else 0.0

        # Normalize timestamps
        joint_t = np.array(joint_t) - start_time
        jacobian_t = np.array(jacobian_t) - start_time
        joints = np.column_stack((joint_t, joint_position, joint_velocity, joint_effort))
        jacobian = np.column_stack((jacobian_t, jacobian))

        if len(force_sensor) > 0:
            force_t = np.array(force_t) - start_time
            force_sensor = np.column_stack((force_t, force_sensor))
        else:
            force_sensor = None

        if len(jaw) > 0:
            jaw_t = np.array(jaw_t) - start_time
            jaw = np.squeeze(np.array(jaw))
            jaw = np.column_stack((jaw_t, jaw))
            min_len = min(len(joints), len(jaw))
            jaw, joints = jaw[:min_len, :], joints[:min_len, :]

        # Save CSVs
        base = f"{self.prefix}{self.index}"
        np.savetxt(f"{self.output}/joints/{base}.csv", joints, delimiter=",")
        np.savetxt(f"{self.output}/jacobian/{base}.csv", jacobian, delimiter=",")
        if force_sensor is not None:
            np.savetxt(f"{self.output}/sensor/{base}.csv", force_sensor, delimiter=",")
        if len(jaw) > 0:
            np.savetxt(f"{self.output}/jaw/{base}.csv", jaw, delimiter=",")

        print(f"Wrote out {base}\n")

    def parse_bags(self):
        print("\nParsing ROS2 bags\n")
        folders = [f for f in os.listdir(self.folder) if os.path.isdir(os.path.join(self.folder, f))]
        folders.sort()
        for folder in folders:
            if os.path.exists(os.path.join(self.folder, folder, "metadata.yaml")):
                self.single_datapoint_processing(os.path.join(self.folder, folder))
                self.index += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', required=True, help='Path to folder containing ROS2 bags')
    parser.add_argument('-o', '--output', required=True, help='Path to write parsed CSVs')
    parser.add_argument('--prefix', default='ros2_', help='Prefix for output files')
    parser.add_argument('--index', default=0, type=int, help='Starting index')
    args = parser.parse_args()

    start = time.time()
    Rosbag2Parser(args).parse_bags()
    print("Parsing complete")
    print(f"Total time: {time.time() - start:.2f} s")


if __name__ == "__main__":
    main()