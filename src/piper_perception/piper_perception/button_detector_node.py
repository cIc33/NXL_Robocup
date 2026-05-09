#!/usr/bin/env python3

from typing import Optional, Tuple

import cv2
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped, PoseStamped
import message_filters
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32
from tf2_geometry_msgs import do_transform_point  # noqa: F401
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker


class ButtonDetectorNode(Node):
    """Detect a red emergency stop button and publish its 3D pose."""

    def __init__(self) -> None:
        super().__init__('button_detector_node')

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._last_camera_info: Optional[CameraInfo] = None
        self._marker_visible = False
        self._warn_timestamps = {}

        self._declare_parameters()
        self._load_parameters()

        self.pose_pub = self.create_publisher(PoseStamped, '/button_pose', 10)
        self.marker_pub = self.create_publisher(Marker, '/button_marker', 10)
        self.depth_pub = self.create_publisher(Float32, '/button_depth_z', 10)
        self.point_camera_pub = self.create_publisher(PointStamped, '/button_point_camera', 10)

        self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self._camera_info_callback,
            qos_profile_sensor_data,
        )

        self.rgb_sub = message_filters.Subscriber(
            self,
            Image,
            self.rgb_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.depth_sub = message_filters.Subscriber(
            self,
            Image,
            self.depth_topic,
            qos_profile=qos_profile_sensor_data,
        )

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
        )
        self.sync.registerCallback(self._image_callback)

        self.get_logger().info('Button detector node ready')

    def _declare_parameters(self) -> None:
        self.declare_parameter('rgb_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_rect_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('camera_frame', '')
        self.declare_parameter('min_area', 500.0)
        self.declare_parameter('min_depth', 0.05)
        self.declare_parameter('max_depth', 5.0)
        self.declare_parameter('lower_red_1', [0, 120, 70])
        self.declare_parameter('upper_red_1', [10, 255, 255])
        self.declare_parameter('lower_red_2', [170, 120, 70])
        self.declare_parameter('upper_red_2', [180, 255, 255])
        self.declare_parameter('marker_scale', 0.03)
        self.declare_parameter('sync_queue_size', 10)
        self.declare_parameter('sync_slop', 0.1)
        self.declare_parameter('gripper_qx', 0.0)
        self.declare_parameter('gripper_qy', -0.707)
        self.declare_parameter('gripper_qz', 0.0)
        self.declare_parameter('gripper_qw', 0.707)

    def _load_parameters(self) -> None:
        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.target_frame = self.get_parameter('target_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.min_area = float(self.get_parameter('min_area').value)
        self.min_depth = float(self.get_parameter('min_depth').value)
        self.max_depth = float(self.get_parameter('max_depth').value)
        self.lower_red_1 = np.array(self.get_parameter('lower_red_1').value, dtype=np.uint8)
        self.upper_red_1 = np.array(self.get_parameter('upper_red_1').value, dtype=np.uint8)
        self.lower_red_2 = np.array(self.get_parameter('lower_red_2').value, dtype=np.uint8)
        self.upper_red_2 = np.array(self.get_parameter('upper_red_2').value, dtype=np.uint8)
        self.marker_scale = float(self.get_parameter('marker_scale').value)
        self.sync_queue_size = int(self.get_parameter('sync_queue_size').value)
        self.sync_slop = float(self.get_parameter('sync_slop').value)
        self.gripper_orientation = {
            'x': float(self.get_parameter('gripper_qx').value),
            'y': float(self.get_parameter('gripper_qy').value),
            'z': float(self.get_parameter('gripper_qz').value),
            'w': float(self.get_parameter('gripper_qw').value),
        }

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        self._last_camera_info = msg

    def _image_callback(self, rgb_msg: Image, depth_msg: Image) -> None:
        if self._last_camera_info is None:
            self._warn_throttled('camera_info', 'Waiting for camera info')
            return

        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except CvBridgeError as exc:
            self._warn_throttled('cv_bridge', f'Failed to convert images: {exc}')
            return

        result = self._detect_button(rgb_image, depth_image)
        if result is None:
            self._delete_marker(rgb_msg.header.stamp)
            return

        u, v, z_m = result
        point_camera = self._project_to_3d(u, v, z_m)
        source_frame = self._resolve_source_frame(rgb_msg, depth_msg)

        point_stamped = PointStamped()
        point_stamped.header = rgb_msg.header
        point_stamped.header.frame_id = source_frame
        point_stamped.point.x = point_camera[0]
        point_stamped.point.y = point_camera[1]
        point_stamped.point.z = point_camera[2]

        depth_msg = Float32()
        depth_msg.data = z_m
        self.depth_pub.publish(depth_msg)
        self.point_camera_pub.publish(point_stamped)

        try:
            transformed = self.tf_buffer.transform(
                point_stamped,
                self.target_frame,
                timeout=Duration(seconds=0.1),
            )
        except TransformException as exc:
            self._warn_throttled('tf', f'Failed to transform button pose: {exc}')
            self._delete_marker(rgb_msg.header.stamp)
            return

        pose = PoseStamped()
        pose.header.stamp = rgb_msg.header.stamp
        pose.header.frame_id = self.target_frame
        pose.pose.position = transformed.point
        pose.pose.orientation.x = self.gripper_orientation['x']
        pose.pose.orientation.y = self.gripper_orientation['y']
        pose.pose.orientation.z = self.gripper_orientation['z']
        pose.pose.orientation.w = self.gripper_orientation['w']

        marker = Marker()
        marker.header = pose.header
        marker.ns = 'button_detector'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = pose.pose
        marker.scale.x = self.marker_scale
        marker.scale.y = self.marker_scale
        marker.scale.z = self.marker_scale
        marker.color.a = 0.9
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        self.pose_pub.publish(pose)
        self.marker_pub.publish(marker)
        self._marker_visible = True

    def _detect_button(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
    ) -> Optional[Tuple[int, int, float]]:
        """Return the button center (u, v) and depth in meters."""
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

        mask_1 = cv2.inRange(hsv_image, self.lower_red_1, self.upper_red_1)
        mask_2 = cv2.inRange(hsv_image, self.lower_red_2, self.upper_red_2)
        mask = cv2.bitwise_or(mask_1, mask_2)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < self.min_area:
            return None

        moments = cv2.moments(contour)
        if moments['m00'] == 0.0:
            return None

        u = int(moments['m10'] / moments['m00'])
        v = int(moments['m01'] / moments['m00'])
        x, y, w, h = cv2.boundingRect(contour)

        depth_m = self._extract_depth(depth_image, u, v, x, y, w, h)
        if depth_m is None:
            return None

        return u, v, depth_m

    def _extract_depth(
        self,
        depth_image: np.ndarray,
        u: int,
        v: int,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> Optional[float]:
        """Read the center depth, falling back to the median inside the box."""
        depth_meters = self._depth_to_meters(depth_image)

        if v < 0 or v >= depth_meters.shape[0] or u < 0 or u >= depth_meters.shape[1]:
            return None

        center_depth = float(depth_meters[v, u])
        if self._is_valid_depth(center_depth):
            return center_depth

        roi = depth_meters[y:y + h, x:x + w]
        valid = roi[np.isfinite(roi) & (roi > 0.0)]
        valid = valid[(valid >= self.min_depth) & (valid <= self.max_depth)]
        if valid.size == 0:
            return None

        return float(np.median(valid))

    def _depth_to_meters(self, depth_image: np.ndarray) -> np.ndarray:
        """Convert depth image to meters while keeping the original layout."""
        if depth_image.dtype == np.uint16:
            return depth_image.astype(np.float32) * 0.001
        return depth_image.astype(np.float32)

    def _is_valid_depth(self, depth_m: float) -> bool:
        return np.isfinite(depth_m) and self.min_depth <= depth_m <= self.max_depth

    def _project_to_3d(self, u: int, v: int, z_m: float) -> Tuple[float, float, float]:
        """Project the detected pixel into the camera optical frame."""
        assert self._last_camera_info is not None
        fx = self._last_camera_info.k[0]
        fy = self._last_camera_info.k[4]
        cx = self._last_camera_info.k[2]
        cy = self._last_camera_info.k[5]

        x = (float(u) - cx) * z_m / fx
        y = (float(v) - cy) * z_m / fy
        return x, y, z_m

    def _resolve_source_frame(self, rgb_msg: Image, depth_msg: Image) -> str:
        if self.camera_frame:
            return self.camera_frame
        if rgb_msg.header.frame_id:
            return rgb_msg.header.frame_id
        if depth_msg.header.frame_id:
            return depth_msg.header.frame_id
        if self._last_camera_info is not None and self._last_camera_info.header.frame_id:
            return self._last_camera_info.header.frame_id
        return 'camera_color_optical_frame'

    def _delete_marker(self, stamp) -> None:
        if not self._marker_visible:
            return

        marker = Marker()
        marker.header.frame_id = self.target_frame
        marker.header.stamp = stamp
        marker.ns = 'button_detector'
        marker.id = 0
        marker.action = Marker.DELETE
        self.marker_pub.publish(marker)
        self._marker_visible = False

    def _warn_throttled(self, key: str, message: str, period: float = 2.0) -> None:
        now = self.get_clock().now().nanoseconds / 1e9
        last_time = self._warn_timestamps.get(key, 0.0)
        if now - last_time >= period:
            self.get_logger().warn(message)
            self._warn_timestamps[key] = now


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ButtonDetectorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
