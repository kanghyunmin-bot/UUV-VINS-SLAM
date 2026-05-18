#!/usr/bin/env python
from __future__ import print_function

import argparse
import csv
import os

import cv2
import rosbag
import rospy
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Header


def main():
    args = parse_args()
    times = read_times(os.path.join(args.dataset_dir, "times.txt"))
    imu_rows = read_imu(os.path.join(args.dataset_dir, "imu.csv"))
    if not times:
        raise SystemExit("times.txt is empty")
    if not imu_rows:
        raise SystemExit("imu.csv is empty")

    output_dir = os.path.dirname(os.path.abspath(args.output_bag))
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if os.path.exists(args.output_bag) and not args.force:
        raise SystemExit("Output bag already exists: %s. Use --force to overwrite." % args.output_bag)
    if os.path.exists(args.output_bag):
        os.remove(args.output_bag)

    with rosbag.Bag(args.output_bag, "w", compression=args.compression) as bag:
        imu_index = 0
        for index, image_time in enumerate(times):
            while imu_index < len(imu_rows) and imu_rows[imu_index][0] <= image_time:
                stamp = rospy.Time.from_sec(imu_rows[imu_index][0] + args.bag_time_offset_sec)
                bag.write(args.imu_topic, make_imu_msg(args.imu_frame, stamp, imu_rows[imu_index]), stamp)
                imu_index += 1

            left_path = os.path.join(args.dataset_dir, "image_0", "%06d.png" % index)
            right_path = os.path.join(args.dataset_dir, "image_1", "%06d.png" % index)
            left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
            right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
            if left is None or right is None:
                raise SystemExit("Missing stereo image pair at index %d" % index)
            stamp = rospy.Time.from_sec(image_time + args.bag_time_offset_sec)
            bag.write(args.image0_topic, make_image_msg(args.image0_frame, stamp, left), stamp)
            bag.write(args.image1_topic, make_image_msg(args.image1_frame, stamp, right), stamp)

        while imu_index < len(imu_rows):
            stamp = rospy.Time.from_sec(imu_rows[imu_index][0] + args.bag_time_offset_sec)
            bag.write(args.imu_topic, make_imu_msg(args.imu_frame, stamp, imu_rows[imu_index]), stamp)
            imu_index += 1

    print(args.output_bag)


def parse_args():
    parser = argparse.ArgumentParser(description="Write exported stereo+IMU VINS dataset to a ROS1 bag.")
    parser.add_argument("dataset_dir")
    parser.add_argument("--output-bag", required=True)
    parser.add_argument("--imu-topic", default="/imu0")
    parser.add_argument("--image0-topic", default="/camera/infra1/image_rect_raw")
    parser.add_argument("--image1-topic", default="/camera/infra2/image_rect_raw")
    parser.add_argument("--imu-frame", default="imu")
    parser.add_argument("--image0-frame", default="infra1")
    parser.add_argument("--image1-frame", default="infra2")
    parser.add_argument("--compression", choices=("none", "bz2", "lz4"), default="none")
    parser.add_argument("--bag-time-offset-sec", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_times(path):
    with open(path) as file:
        return [float(line.strip()) for line in file if line.strip()]


def read_imu(path):
    rows = []
    with open(path) as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                (
                    float(row["timestamp_sec"]),
                    float(row["ax"]),
                    float(row["ay"]),
                    float(row["az"]),
                    float(row["gx"]),
                    float(row["gy"]),
                    float(row["gz"]),
                )
            )
    return rows


def make_imu_msg(frame_id, stamp, row):
    timestamp_sec, ax, ay, az, gx, gy, gz = row
    msg = Imu()
    msg.header = Header()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.linear_acceleration.x = ax
    msg.linear_acceleration.y = ay
    msg.linear_acceleration.z = az
    msg.angular_velocity.x = gx
    msg.angular_velocity.y = gy
    msg.angular_velocity.z = gz
    msg.orientation_covariance[0] = -1.0
    return msg


def make_image_msg(frame_id, stamp, image):
    msg = Image()
    msg.header = Header()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = image.shape[0]
    msg.width = image.shape[1]
    msg.encoding = "mono8"
    msg.is_bigendian = 0
    msg.step = image.shape[1]
    msg.data = image.tobytes() if hasattr(image, "tobytes") else image.tostring()
    return msg


if __name__ == "__main__":
    main()
