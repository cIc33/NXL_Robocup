import cv2
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


class GoProPublisher(Node):
    def __init__(self):
        super().__init__('gopro_camera')
        self.pub = self.create_publisher(CompressedImage, '/gopro/image_raw/compressed', 10)

        self.cap = cv2.VideoCapture('/dev/video42', cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YU12'))
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Warm up: descarta frames iniciales sucios
        for _ in range(5):
            self.cap.grab()

        self.lock = threading.Lock()
        self.frame_data = None
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        self.timer = self.create_timer(1/30.0, self.publish_frame)

    def _capture_loop(self):
        while rclpy.ok():
            # grab() descarta el frame en buffer (puede estar incompleto)
            self.cap.grab()
            # retrieve() lee el siguiente ya completo
            ret, frame = self.cap.retrieve()
            if ret:
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                with self.lock:
                    self.frame_data = jpeg.tobytes()

    def publish_frame(self):
        with self.lock:
            if self.frame_data is None:
                return
            data = self.frame_data

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = 'jpeg'
        msg.data = bytes(data)
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = GoProPublisher()
    rclpy.spin(node)
    node.cap.release()
    node.destroy_node()
    rclpy.shutdown()