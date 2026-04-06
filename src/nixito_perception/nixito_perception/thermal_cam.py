import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import numpy as np
import board
import busio
import adafruit_mlx90640


class ThermalPublisher(Node):
    def __init__(self):
        super().__init__('thermal_publisher')

        self.publisher_ = self.create_publisher(Image, '/thermal/image', 10)
        self.bridge = CvBridge()

        # Inicializa el sensor MLX90640
        self.mlx = self._initialize_sensor()

        if self.mlx is not None:
            self.get_logger().info('Sensor MLX90640 inicializado correctamente.')
            # Publica a 16 Hz para coincidir con el refresh rate del sensor
            self.timer = self.create_timer(1.0 / 16.0, self.publish_thermal)
        else:
            self.get_logger().error('No se pudo inicializar el sensor MLX90640. Abortando.')

    def _initialize_sensor(self):
        """Inicializa el bus I2C y el sensor térmico MLX90640."""
        try:
            i2c = busio.I2C(board.SCL, board.SDA)
            mlx = adafruit_mlx90640.MLX90640(i2c)
            mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ
            return mlx
        except Exception as e:
            self.get_logger().error(f'Error al inicializar el sensor: {e}')
            return None

    def _get_thermal_frame(self):
        """Lee un frame térmico del sensor MLX90640.
        
        Returns:
            np.ndarray de forma (24, 32) con temperaturas en °C, o None si falla.
        """
        raw = np.zeros((24 * 32,), dtype=np.float32)
        try:
            self.mlx.getFrame(raw)
        except ValueError as e:
            self.get_logger().warn(f'Error al leer frame térmico: {e}')
            return None
        return np.reshape(raw, (24, 32)).astype(np.float64)

    def publish_thermal(self):
        """Callback del timer: lee el sensor y publica la imagen térmica."""
        frame = self._get_thermal_frame()

        if frame is None:
            self.get_logger().warn('Frame inválido, se omite esta publicación.')
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='64FC1')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'thermal_frame'

        self.publisher_.publish(msg)
        self.get_logger().debug(
            f'Frame publicado | min={frame.min():.1f}°C  max={frame.max():.1f}°C'
        )


def main():
    rclpy.init()
    node = ThermalPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()