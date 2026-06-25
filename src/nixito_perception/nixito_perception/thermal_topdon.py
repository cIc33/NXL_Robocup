import cv2
import numpy as np
import rclpy
import os
import glob
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def find_tc001_device():
    """
    Busca el dispositivo /dev/videoX de la cámara Topdon TC001
    comparando Vendor ID (0x0bda) y Product ID (0x5830) en sysfs.
    Retorna la ruta del dispositivo o None si no se encuentra.
    """
    TC001_VENDOR_ID = '0bda'
    TC001_PRODUCT_ID = '5830'

    for video_path in sorted(glob.glob('/sys/class/video4linux/video*')):
        try:
            real_path = os.path.realpath(video_path)
            parts = real_path.split('/')
            for i in range(len(parts), 0, -1):
                parent = '/'.join(parts[:i])
                vendor_file = os.path.join(parent, 'idVendor')
                product_file = os.path.join(parent, 'idProduct')
                if os.path.exists(vendor_file) and os.path.exists(product_file):
                    with open(vendor_file) as vf, open(product_file) as pf:
                        vendor = vf.read().strip()
                        product = pf.read().strip()
                    if vendor == TC001_VENDOR_ID and product == TC001_PRODUCT_ID:
                        dev_name = os.path.basename(video_path)
                        return f'/dev/{dev_name}'
                    break
        except (OSError, PermissionError):
            continue

    return None


class TC001ThermalNode(Node):
    def __init__(self):
        super().__init__('tc001_thermal_node')

        # Parámetros de imagen
        self.width = 256
        self.height = 192
        self.scale = 3
        self.new_width = self.width * self.scale
        self.new_height = self.height * self.scale

        # Publisher y bridge
        self.publisher = self.create_publisher(Image, 'thermal/image_raw', 10)
        self.bridge = CvBridge()

        # Detección automática con opción de override por parámetro ROS
        self.declare_parameter('device', '')
        device_param = self.get_parameter('device').get_parameter_value().string_value

        if device_param:
            device_path = device_param
            self.get_logger().info(f'Usando dispositivo por parámetro: {device_path}')
        else:
            device_path = find_tc001_device()
            if device_path:
                self.get_logger().info(f'TC001 detectada automáticamente en: {device_path}')
            else:
                self.get_logger().error(
                    'No se encontró la cámara TC001. '
                    'Verifica que esté conectada o usa: --ros-args -p device:=/dev/videoX'
                )
                raise RuntimeError('Cámara TC001 no encontrada')

        # Abrir captura
        self.cap = cv2.VideoCapture(device_path, cv2.CAP_V4L)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)

        if not self.cap.isOpened():
            self.get_logger().error(f'No se pudo abrir {device_path}')
            raise RuntimeError(f'No se pudo abrir la cámara TC001 en {device_path}')

        # Timer a ~25 Hz
        self.timer = self.create_timer(0.04, self.timer_callback)
        self.get_logger().info('TC001 Thermal Node iniciado. Publicando en: thermal/image_raw')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('No se recibió frame de la cámara')
            return

        # Separar imagen visible y datos de temperatura
        imdata, thdata = np.array_split(frame, 2)

        # --- Temperatura del centro ---
        hi = thdata[96][128][0]
        lo = thdata[96][128][1]
        rawtemp = hi + (lo * 256)
        center_temp = round((rawtemp / 64) - 273.15, 2)

        # --- Temperatura máxima ---
        lomax = thdata[..., 1].max()
        posmax = thdata[..., 1].argmax()
        mcol, mrow = divmod(posmax, self.width)
        himax = thdata[mcol][mrow][0]
        maxtemp = round(((himax + lomax * 256) / 64) - 273.15, 2)

        # --- Temperatura mínima ---
        lomin = thdata[..., 1].min()
        posmin = thdata[..., 1].argmin()
        lcol, lrow = divmod(posmin, self.width)
        himin = thdata[lcol][lrow][0]
        mintemp = round(((himin + lomin * 256) / 64) - 273.15, 2)

        # --- Temperatura promedio (threshold para puntos) ---
        loavg = thdata[..., 1].mean()
        hiavg = thdata[..., 0].mean()
        avgtemp = round(((loavg * 256 + hiavg) / 64) - 273.15, 2)
        threshold = 2  # °C sobre/bajo el promedio para mostrar punto

        # --- Procesar imagen visual ---
        bgr = cv2.cvtColor(imdata, cv2.COLOR_YUV2BGR_YUYV)
        bgr = cv2.resize(bgr, (self.new_width, self.new_height), interpolation=cv2.INTER_CUBIC)
        heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_INFERNO)

        cx = self.new_width // 2
        cy = self.new_height // 2

        # --- Cruz central ---
        cv2.line(heatmap, (cx, cy + 20), (cx, cy - 20), (255, 255, 255), 2)
        cv2.line(heatmap, (cx + 20, cy), (cx - 20, cy), (255, 255, 255), 2)
        cv2.line(heatmap, (cx, cy + 20), (cx, cy - 20), (0, 0, 0), 1)
        cv2.line(heatmap, (cx + 20, cy), (cx - 20, cy), (0, 0, 0), 1)

        # Temperatura centro
        cv2.putText(heatmap, f'{center_temp} C', (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(heatmap, f'{center_temp} C', (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        # --- Punto más caliente ---
        if maxtemp > avgtemp + threshold:
            px, py = mrow * self.scale, mcol * self.scale
            cv2.circle(heatmap, (px, py), 5, (0, 0, 0), 2)
            cv2.circle(heatmap, (px, py), 5, (0, 0, 255), -1)
            cv2.putText(heatmap, f'{maxtemp} C', (px + 10, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(heatmap, f'{maxtemp} C', (px + 10, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        # --- Punto más frío ---
        if mintemp < avgtemp - threshold:
            px, py = lrow * self.scale, lcol * self.scale
            cv2.circle(heatmap, (px, py), 5, (0, 0, 0), 2)
            cv2.circle(heatmap, (px, py), 5, (255, 0, 0), -1)
            cv2.putText(heatmap, f'{mintemp} C', (px + 10, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(heatmap, f'{mintemp} C', (px + 10, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        # --- Publicar imagen ---
        msg = self.bridge.cv2_to_imgmsg(heatmap, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'thermal_camera'
        self.publisher.publish(msg)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TC001ThermalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()