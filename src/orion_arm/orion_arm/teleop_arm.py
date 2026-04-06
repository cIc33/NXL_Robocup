import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8MultiArray, Float32MultiArray
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import serial
import math


class ControlTest(Node):
    def __init__(self):
        super().__init__('control_test')

        qos_cmds = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.pub_brazo = self.create_publisher(Int8MultiArray, '/brazo/raw_cmd', qos_cmds)
        self.pub_servos = self.create_publisher(Int8MultiArray, '/brazo/servos_cmd', 10)
        self.pub_encoders = self.create_publisher(Float32MultiArray, '/nixito/encoders', 10)
        self.pub_velocidad = self.create_publisher(Float32MultiArray, '/nixito/velocidad', 10)

        # NUEVO: publisher para URDF/TF
        self.pub_joint_states = self.create_publisher(JointState, '/nixito/joint_states', 10)

        try:
            self.ser = serial.Serial('/dev/ttyTHS1', 115200, timeout=0.1)
            self.get_logger().info("Conectado al Arduino en /dev/ttyTHS1 a 115200 baudios")
        except Exception as e:
            self.get_logger().error(f"Error crítico: No se pudo abrir /dev/ttyTHS1: {e}")
            raise e

        self.serial_buffer = ""
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

                self.get_logger().info(f"SERIAL RAW: {line}")

                if line.startswith("DRV:"):
                    self.procesar_drv(line)
                elif line.startswith("ARM:"):
                    self.procesar_arm(line)

        except Exception as e:
            self.get_logger().error(f"Error procesando serial: {e}")

    def procesar_drv(self, line):
        try:
            datos = line.replace("DRV:", "").split(',')

            if len(datos) != 4:
                self.get_logger().warn(f"DRV invalido: {datos}")
                return

            values = [float(val) for val in datos]

            encoders = values[:2]
            velocidad = values[2:]

            self.get_logger().info(f"ENC={encoders} VEL={velocidad}")

            msg_enc = Float32MultiArray()
            msg_enc.data = encoders
            self.pub_encoders.publish(msg_enc)

            msg_vel = Float32MultiArray()
            msg_vel.data = velocidad
            self.pub_velocidad.publish(msg_vel)

            # NUEVO: publicar solo la parte de tracción en /joint_states
            msg_js = JointState()
            msg_js.header.stamp = self.get_clock().now().to_msg()
            msg_js.name = [
                'dir_R_joint',
                'dir_L_joint'
            ]

            # AJUSTA ESTA CONVERSION SEGUN TU UNIDAD REAL:
            # - si ya vienen en radianes: usa encoders directo
            # - si vienen en grados: usa math.radians(...)
            # - si vienen en cuentas: convierte cuentas -> radianes
            msg_js.position = [
                encoders[0],
                encoders[1]
            ]

            self.pub_joint_states.publish(msg_js)

        except ValueError as e:
            self.get_logger().warn(f"DRV valor no numerico: {e}")

    def procesar_arm(self, line):
        try:
            parts = line.replace("ARM:", "").split(',')

            if len(parts) != 6:
                self.get_logger().warn(f"ARM invalido: {parts}")
                return

            raw_values = [int(val) for val in parts]

            brazo = [int(val / 10.0) for val in raw_values[:4]]
            servos = raw_values[4:6]

            self.get_logger().info(f"BRAZO={brazo} SERVOS={servos}")

            msg_brazo = Int8MultiArray()
            msg_brazo.data = brazo
            self.pub_brazo.publish(msg_brazo)

            msg_servos = Int8MultiArray()
            msg_servos.data = servos
            self.pub_servos.publish(msg_servos)

        except ValueError as e:
            self.get_logger().warn(f"ARM valor no numerico: {e}")

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