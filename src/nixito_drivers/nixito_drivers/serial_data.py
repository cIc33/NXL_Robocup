import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8MultiArray, Float32MultiArray
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import serial


class ControlTest(Node):
    def __init__(self):
        super().__init__('control_test')

        qos_cmds = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.pub_brazo = self.create_publisher(Int8MultiArray, '/brazo/raw_cmd', qos_cmds)
        self.pub_servos = self.create_publisher(Int8MultiArray, '/brazo/servos_cmd', qos_cmds)

        # JointState para encoders de tracción: position = pulsos, velocity = QPPS
        self.pub_encoders = self.create_publisher(JointState, '/nixito/drive_encoders', qos_cmds)
        self.create_subscription(
            Float32MultiArray,
            '/cmd_vel',
            self._update_tx_values,
            qos_cmds
        )

        self.tx_values = [0, 0]
        self.serial_buffer = ""

        try:
            self.ser = serial.Serial('/dev/ttyTHS1', 115200, timeout=0.01)
            self.get_logger().info("Conectado al Arduino en /dev/ttyTHS1 a 115200 baudios")
        except Exception as e:
            self.get_logger().error(f"Error crítico: No se pudo abrir /dev/ttyTHS1: {e}")
            raise e

        self.timer = self.create_timer(0.02, self.timer_callback)

    def timer_callback(self):
        try:
            if self.ser.in_waiting > 0:
                raw = self.ser.read(self.ser.in_waiting)
                self.serial_buffer += raw.decode('utf-8', errors='ignore')

            while '\n' in self.serial_buffer:
                line, self.serial_buffer = self.serial_buffer.split('\n', 1)
                line = line.strip()

                if not line:
                    continue

                if line.startswith("ARM:"):
                    self._parse_arm(line)
                elif line.startswith("DRV:"):
                    self._parse_drv(line)

            self._write_tx_values()

        except Exception as e:
            self.get_logger().error(f"Error procesando serial: {e}")

    def _update_tx_values(self, msg):
        data = list(msg.data)

        if len(data) < 2:
            self.get_logger().warn(f"/tal_topico invalido, se esperaban 2 valores: {data}")
            return

        self.tx_values = [int(data[0]), int(data[1])]

        self.get_logger().info(f"{data[0]},{data[1]}")

    def _write_tx_values(self):
        payload = f"{self.tx_values[0]},{self.tx_values[1]}\n"

        try:
            self.ser.write(payload.encode('utf-8'))
        except Exception as e:
            self.get_logger().error(f"Error enviando serial '{payload.strip()}': {e}")

    def _parse_arm(self, line):
        parts = line.replace("ARM:", "").split(',')

        if len(parts) != 6:
            return

        raw_values = [int(val) for val in parts]

        # Escala de -100..100 a -10..10
        brazo = [int(val / 10.0) for val in raw_values[:4]]
        servos = raw_values[4:6]

        msg_brazo = Int8MultiArray()
        msg_brazo.data = brazo
        self.pub_brazo.publish(msg_brazo)

        msg_servos = Int8MultiArray()
        msg_servos.data = servos
        self.pub_servos.publish(msg_servos)



    def _parse_drv(self, line):
        parts = line.replace("DRV:", "").split(',')

        if len(parts) != 4:
            self.get_logger().warn(f"DRV invalido: {parts}")
            return

        # DRV:pos_izq, pos_der, vel_izq, vel_der
        pos_izq = float(parts[0])
        pos_der = float(parts[1])
        vel_izq = float(parts[2])
        vel_der = float(parts[3])

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['drive_left', 'drive_right']
        msg.position = [pos_izq, pos_der]   # pulsos acumulados del encoder
        msg.velocity = [vel_izq, vel_der]   # QPPS (quadrature pulses per second)
        msg.effort = []

        self.pub_encoders.publish(msg)



def main(args=None):
    rclpy.init(args=args)
    node = ControlTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Deteniendo nodo por teclado...")
    finally:
        if hasattr(node, 'ser') and node.ser.is_open:
            node.ser.close()
            node.get_logger().info("Puerto serial cerrado.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
