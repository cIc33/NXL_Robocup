# En tu paquete nixito_perception, agregar un script: gopro_publisher.py
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class GoProPublisher(Node):
    def __init__(self):
        super().__init__('gopro_camera')
        self.pub = self.create_publisher(Image, '/image_raw', 10)
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture('/dev/video42')
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.timer = self.create_timer(1/30.0, self.publish_frame)

    def publish_frame(self):
        ret, frame = self.cap.read()
        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            msg.header.stamp = self.get_clock().now().to_msg()
            self.pub.publish(msg)

def main():
    rclpy.init()
    node = GoProPublisher()
    rclpy.spin(node)
    node.cap.release()
    rclpy.shutdown()