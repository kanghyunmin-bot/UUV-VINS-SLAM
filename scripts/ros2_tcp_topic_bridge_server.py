#!/usr/bin/env python3
import argparse
import json
import socket
import struct
import threading
from typing import Dict, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.serialization import serialize_message
from rosidl_runtime_py.utilities import get_message


def parse_topic(spec: str) -> Tuple[str, str]:
    if ":" not in spec:
        raise argparse.ArgumentTypeError(f"topic spec must be /topic:pkg/msg/Type, got {spec}")
    topic, type_name = spec.split(":", 1)
    if not topic.startswith("/"):
        raise argparse.ArgumentTypeError(f"topic must be absolute, got {topic}")
    return topic, type_name


class TcpTopicBridgeServer(Node):
    def __init__(self, host: str, port: int, topics: Dict[str, str], queue_size: int):
        super().__init__("uuv_tcp_topic_bridge_server")
        self._host = host
        self._port = port
        self._topics = topics
        self._topic_order = list(topics.keys())
        self._latest: Dict[str, bytes] = {}
        self._tf_static_by_child = {}
        self._condition = threading.Condition()
        self._subscriptions = []

        for topic, type_name in topics.items():
            msg_type = get_message(type_name)
            qos = QoSProfile(depth=queue_size)
            if topic == "/tf_static":
                qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
            self._subscriptions.append(
                self.create_subscription(
                    msg_type,
                    topic,
                    lambda msg, topic=topic, type_name=type_name: self._enqueue(topic, type_name, msg),
                    qos,
                )
            )

        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        self.get_logger().info(f"TCP topic bridge serving {len(topics)} topics on {host}:{port}")

    def _enqueue(self, topic: str, type_name: str, msg) -> None:
        if topic == "/tf_static" and hasattr(msg, "transforms"):
            for transform in msg.transforms:
                self._tf_static_by_child[transform.child_frame_id] = transform
            combined = type(msg)()
            combined.transforms = list(self._tf_static_by_child.values())
            msg = combined
        header = json.dumps({"topic": topic, "type": type_name}, separators=(",", ":")).encode("utf-8")
        payload = bytes(serialize_message(msg))
        packet = struct.pack("!I", len(header)) + header + struct.pack("!I", len(payload)) + payload
        with self._condition:
            self._latest[topic] = packet
            self._condition.notify()

    def _take_latest_packets(self) -> Dict[str, bytes]:
        with self._condition:
            while rclpy.ok() and not self._latest:
                self._condition.wait(timeout=0.5)
            packets = self._latest
            self._latest = {}
            return packets

    def _serve(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self._host, self._port))
        server.listen(1)
        while rclpy.ok():
            conn = None
            try:
                conn, addr = server.accept()
                self.get_logger().info(f"TCP bridge client connected: {addr}")
                with conn:
                    while rclpy.ok():
                        latest_packets = self._take_latest_packets()
                        for topic in self._topic_order:
                            packet = latest_packets.get(topic)
                            if packet is not None:
                                conn.sendall(packet)
            except OSError as exc:
                self.get_logger().warn(f"TCP bridge client disconnected: {exc}")
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except OSError:
                        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--queue-size", type=int, default=300)
    parser.add_argument("--topic", action="append", type=parse_topic, required=True)
    args = parser.parse_args()

    topics = dict(args.topic)
    rclpy.init()
    node = TcpTopicBridgeServer(args.host, args.port, topics, args.queue_size)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
