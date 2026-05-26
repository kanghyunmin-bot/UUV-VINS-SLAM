#!/usr/bin/env python3
import argparse
import json
import socket
import struct
import time
from typing import Dict, Type

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def read_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        data = sock.recv(remaining)
        if not data:
            raise ConnectionError("socket closed")
        chunks.append(data)
        remaining -= len(data)
    return b"".join(chunks)


class TcpTopicBridgeClient(Node):
    def __init__(self, host: str, port: int, reconnect_sec: float):
        super().__init__("uuv_tcp_topic_bridge_client")
        self._host = host
        self._port = port
        self._reconnect_sec = reconnect_sec
        self._topic_publishers: Dict[str, object] = {}
        self._types: Dict[str, Type] = {}

    def _publisher_for(self, topic: str, type_name: str):
        if topic not in self._topic_publishers:
            msg_type = get_message(type_name)
            self._types[topic] = msg_type
            qos = QoSProfile(depth=10)
            if topic.startswith("/dvl_reference") or topic in {"/path", "/odometry", "/tf_static"}:
                qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
            self._topic_publishers[topic] = self.create_publisher(msg_type, topic, qos)
            self.get_logger().info(f"republishing {topic} as {type_name}")
        return self._topic_publishers[topic], self._types[topic]

    def run(self) -> None:
        while rclpy.ok():
            try:
                self.get_logger().info(f"connecting TCP topic bridge {self._host}:{self._port}")
                with socket.create_connection((self._host, self._port), timeout=5.0) as sock:
                    sock.settimeout(None)
                    self.get_logger().info("TCP topic bridge connected")
                    while rclpy.ok():
                        header_len = struct.unpack("!I", read_exact(sock, 4))[0]
                        header = json.loads(read_exact(sock, header_len).decode("utf-8"))
                        payload_len = struct.unpack("!I", read_exact(sock, 4))[0]
                        payload = read_exact(sock, payload_len)
                        pub, msg_type = self._publisher_for(header["topic"], header["type"])
                        pub.publish(deserialize_message(payload, msg_type))
                        rclpy.spin_once(self, timeout_sec=0.0)
            except Exception as exc:
                self.get_logger().warn(f"TCP topic bridge reconnecting after: {exc}")
                time.sleep(self._reconnect_sec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--reconnect-sec", type=float, default=1.0)
    args = parser.parse_args()

    rclpy.init()
    node = TcpTopicBridgeClient(args.host, args.port, args.reconnect_sec)
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
