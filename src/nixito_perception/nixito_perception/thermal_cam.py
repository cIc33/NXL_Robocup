import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import serial
import threading

class ThermalPublisher(Node):
    def __init__(self):
        super().__init__('thermal_publisher')
        self.publisher_ = self.create_publisher(Image, '/thermal/image', 10)
        self.bridge = CvBridge()

        # Buffer compartido entre hilos
        self.latest_frame = None
        self.lock = threading.Lock()

        # Inicializa serial
        self.ser = self._initialize_sensor()
        if self.ser is not None:
            self.get_logger().info('ESP32 conectada correctamente.')

            # Hilo de lectura en paralelo
            self.read_thread = threading.Thread(target=self._read_serial_loop, daemon=True)
            self.read_thread.start()

            # Publica a 8Hz (ajusta según tu ESP)
            self.timer = self.create_timer(1.0 / 8.0, self.publish_thermal)
        else:
            self.get_logger().error('No se pudo conectar con la ESP. Abortando.')

    def _initialize_sensor(self):
        """Inicializa la conexión serial con la ESP."""
        try:
            ser = serial.Serial('/dev/ttyCH341USB0', 921600, timeout=2)
            return ser
        except Exception as e:
            self.get_logger().error(f'Error al conectar serial: {e}')
            return None

    def _read_serial_loop(self):
        """Hilo que lee continuamente frames de la ESP."""
        while True:
            try:
                # Sincronizar con marcador 0xAA 0xBB
                while True:
                    if self.ser.read(1) == b'\xAA':
                        if self.ser.read(1) == b'\xBB':
                            break

                # Leer 768 floats (4 bytes cada uno)
                buf = b''
                while len(buf) < 768 * 4:
                    chunk = self.ser.read(768 * 4 - len(buf))
                    if not chunk:
                        break
                    buf += chunk

                if len(buf) == 768 * 4:
                    frame = np.frombuffer(buf, dtype=np.float32).reshape(24, 32).astype(np.float64)
                    with self.lock:
                        self.latest_frame = frame

            except Exception as e:
                self.get_logger().warn(f'Error leyendo serial: {e}')
                continue

    def publish_thermal(self):
        """Callback del timer: publica el último frame recibido."""
        with self.lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None

        if frame is None:
            self.get_logger().warn('Aún no hay frame disponible.')
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
        if node.ser and node.ser.is_open:
            node.ser.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
