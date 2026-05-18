#!/usr/bin/env python
from __future__ import print_function

import argparse
import csv
import os
import time

import cv2
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

    rospy.init_node("vins_stereo_imu_dataset_publisher", anonymous=True)
    pub_left = rospy.Publisher(args.image0_topic, Image, queue_size=100)
    pub_right = rospy.Publisher(args.image1_topic, Image, queue_size=100)
    pub_imu = rospy.Publisher(args.imu_topic, Imu, queue_size=2000)
    wait_for_subscribers([pub_left, pub_right, pub_imu], args.wait_subscribers)

    imu_index = 0
    previous_image_time = times[0]
    for index, image_time in enumerate(times):
        if rospy.is_shutdown():
            break
        imu_publish_until = image_time + args.imu_lead_time
        while imu_index < len(imu_rows) and imu_rows[imu_index][0] <= imu_publish_until:
            pub_imu.publish(make_imu_msg(args.imu_topic, imu_rows[imu_index]))
            imu_index += 1

        left_path = os.path.join(args.dataset_dir, "image_0", "%06d.png" % index)
        right_path = os.path.join(args.dataset_dir, "image_1", "%06d.png" % index)
        left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
        if left is None or right is None:
            raise SystemExit("Missing image pair at index %d" % index)
        stamp = rospy.Time.from_sec(image_time)
        pub_left.publish(make_image_msg(args.image0_frame, stamp, left))
        pub_right.publish(make_image_msg(args.image1_frame, stamp, right))

        dt = max(0.0, image_time - previous_image_time)
        previous_image_time = image_time
        sleep_time = max(args.min_sleep, min(args.max_sleep, dt / max(args.realtime_factor, 1.0)))
        if sleep_time > 0:
            rospy.sleep(sleep_time)

    while imu_index < len(imu_rows) and not rospy.is_shutdown():
        pub_imu.publish(make_imu_msg(args.imu_topic, imu_rows[imu_index]))
        imu_index += 1
    rospy.sleep(args.final_sleep)


def parse_args():
    parser = argparse.ArgumentParser(description="Replay an exported stereo+IMU dataset into VINS-Fusion ROS topics.")
    parser.add_argument("dataset_dir")
    parser.add_argument("--imu-topic", default="/imu0")
    parser.add_argument("--image0-topic", default="/camera/infra1/image_rect_raw")
    parser.add_argument("--image1-topic", default="/camera/infra2/image_rect_raw")
    parser.add_argument("--image0-frame", default="infra1")
    parser.add_argument("--image1-frame", default="infra2")
    parser.add_argument("--wait-subscribers", type=float, default=15.0)
    parser.add_argument("--realtime-factor", type=float, default=8.0)
    parser.add_argument("--imu-lead-time", type=float, default=0.03)
    parser.add_argument("--min-sleep", type=float, default=0.002)
    parser.add_argument("--max-sleep", type=float, default=0.05)
    parser.add_argument("--final-sleep", type=float, default=2.0)
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


def wait_for_subscribers(publishers, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline and not rospy.is_shutdown():
        if all(pub.get_num_connections() > 0 for pub in publishers):
            return
        rospy.sleep(0.1)
    print("subscriber wait timeout; publishing anyway")


def make_imu_msg(frame_id, row):
    timestamp_sec, ax, ay, az, gx, gy, gz = row
    msg = Imu()
    msg.header = Header()
    msg.header.stamp = rospy.Time.from_sec(timestamp_sec)
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
