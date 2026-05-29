import cv2
import threading
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class GoProPublisher(Node):
    def __init__(self):
        super().__init__('gopro_camera')

        # Parámetros configurables desde el launch
        self.declare_parameter('device', '/dev/video42')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30)

        device = self.get_parameter('device').value
        width  = self.get_parameter('width').value
        height = self.get_parameter('height').value
        fps    = self.get_parameter('fps').value

        # QoS Best Effort depth=1: descarta mensajes viejos, nunca acumula
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.pub = self.create_publisher(Image, '/image_raw', qos)
        self.bridge = CvBridge()

        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS,          fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # buffer mínimo

        if not self.cap.isOpened():
            self.get_logger().error(f'No se pudo abrir {device}')
            return

        self.get_logger().info(
            f'GoPro abierta: {device} @ {width}x{height} {fps}fps'
        )

        self._stop = False
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        while not self._stop:
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn(
                    'Frame vacío', throttle_duration_sec=2.0
                )
                continue

            # Resize por si ffmpeg entregó resolución distinta
            h, w = frame.shape[:2]
            target_w = self.get_parameter('width').value
            target_h = self.get_parameter('height').value
            if (w, h) != (target_w, target_h):
                frame = cv2.resize(
                    frame, (target_w, target_h),
                    interpolation=cv2.INTER_LINEAR
                )

            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'gopro_frame'
            self.pub.publish(msg)

    def destroy_node(self):
        self._stop = True
        self._thread.join(timeout=2.0)
        self.cap.release()
        super().destroy_node()


def main():
    rclpy.init()
    node = GoProPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()