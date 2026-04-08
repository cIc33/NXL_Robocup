#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import serial
from std_msgs.msg import Float32MultiArray, Int8MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_srvs.srv import Trigger


class SerialBridge(Node):
    def __init__(self):
        super().__init__('esp32_serial_bridge')

        qos_sensors = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )
        qos_cmds = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Conexión USB con el ESP32
        self.ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.05)

        # Publica los ángulos que lee del ESP32
        self.sensor_pub = self.create_publisher(
            Float32MultiArray, 'brazo/angulos_sensores', qos_sensors
        )

        # Recibe los comandos — ahora Int8MultiArray, sin empaquetado
        self.cmd_sub = self.create_subscription(
            Int8MultiArray, '/brazo/joint_cmds', self.cmd_callback, qos_cmds
        )

        # Servicio de calibración
        self.srv = self.create_service(Trigger, 'brazo/calibrar_zero', self.calibrar_cb)

        # Timer de lectura serial
        self.timer = self.create_timer(0.01, self.read_serial)
        self.get_logger().info('Puente Serial iniciado en /dev/ttyUSB0')

    def cmd_callback(self, msg):
        """
        Recibe [seq, e1, e2, e3, e4] como Int8MultiArray y lo envía
        al ESP32 en texto plano: "JOINT_CMD seq e1 e2 e3 e4\n"
        El ESP32 lo parsea con sscanf y lo reenvía al Arduino como STATE.
        """
        if len(msg.data) < 5:
            self.get_logger().warn(f'joint_cmds incompleto: {list(msg.data)}')
            return

        seq, e1, e2, e3, e4 = msg.data[0], msg.data[1], msg.data[2], msg.data[3], msg.data[4]
        comando_str = f"JOINT_CMD {seq} {e1} {e2} {e3} {e4}\n"

        try:
            self.ser.write(comando_str.encode('utf-8'))
            self.get_logger().info(f'→ ESP32: {comando_str.strip()}')
        except Exception as e:
            self.get_logger().error(f'Error escribiendo serial: {e}')

    def calibrar_cb(self, request, response):
        self.ser.write(b"CALIBRAR\n")
        response.success = True
        response.message = "Comando de calibración enviado"
        return response

    def read_serial(self):
        if self.ser.in_waiting > 0:
            try:
                linea = self.ser.readline().decode('utf-8').strip()

                if linea.startswith("SENS"):
                    partes = linea.split()
                    if len(partes) == 5:
                        msg = Float32MultiArray()
                        msg.data = [float(partes[1]), float(partes[2]),
                                    float(partes[3]), float(partes[4])]
                        self.sensor_pub.publish(msg)
                    else:
                        self.get_logger().warn(f'SENS malformado: "{linea}"')

                elif linea.startswith("INFO"):
                    self.get_logger().info(f'ESP32: {linea}')

            except UnicodeDecodeError:
                pass  # Basura serial al inicio
            except Exception as e:
                self.get_logger().error(f'Error leyendo serial: {e}')


def main(args=None):
    rclpy.init(args=args)
    nodo = SerialBridge()
    try:
        rclpy.spin(nodo)
    finally:
        nodo.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()