import os
import glob
import rosbag


def find_bag_file(folder):
    """Find the first .bag file in a folder."""
    print("FIND BAG FOLDER")
    bags = glob.glob(os.path.join(folder, "*.bag"))
    if not bags:
        raise FileNotFoundError(f"No .bag files found in {folder}")
    return bags[0]  # return the first match


def split_rosbag_by_time(folder, val_name="val.bag", test_name="test.bag"):
    """
    Find the first bag file in a folder and split it into val/test halves by time.
    """

    print("HEREE")
    input_bag = find_bag_file(folder)
    val_path = os.path.join(folder, val_name)
    test_path = os.path.join(folder, test_name)

    with rosbag.Bag(input_bag, 'r') as bag, \
         rosbag.Bag(val_path, 'w') as val_bag, \
         rosbag.Bag(test_path, 'w') as test_bag:

        start_time = bag.get_start_time()
        end_time = bag.get_end_time()
        mid_time = start_time + (end_time - start_time) / 2.0

        for topic, msg, t in bag.read_messages():
            if t.to_sec() <= mid_time:
                val_bag.write(topic, msg, t)
            else:
                test_bag.write(topic, msg, t)

    print(f"[DONE] Split {input_bag} → {val_path}, {test_path}")


def split_rosbag_by_count(folder, val_name="val.bag", test_name="test.bag"):
    """
    Find the first bag file in a folder and split it into val/test halves by message count.
    """
    input_bag = find_bag_file(folder)
    val_path = os.path.join(folder, val_name)
    test_path = os.path.join(folder, test_name)

    # First pass: count messages
    with rosbag.Bag(input_bag, 'r') as bag:
        total_msgs = bag.get_message_count()
    midpoint = total_msgs // 2

    with rosbag.Bag(input_bag, 'r') as bag, \
         rosbag.Bag(val_path, 'w') as val_bag, \
         rosbag.Bag(test_path, 'w') as test_bag:

        for i, (topic, msg, t) in enumerate(bag.read_messages()):
            if i < midpoint:
                val_bag.write(topic, msg, t)
            else:
                test_bag.write(topic, msg, t)

    print(f"[DONE] Split {input_bag} → {val_path}, {test_path}")