import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8MultiArray, Float32MultiArray
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import serial
import time


class ControlTest(Node):
    def __init__(self):
        super().__init__('control_test')

        qos_cmds = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Publishers brazo
        self.pub_brazo = self.create_publisher(Int8MultiArray, '/brazo/raw_cmd', qos_cmds)
        self.pub_servos = self.create_publisher(Int8MultiArray, '/brazo/servos_cmd', qos_cmds)

        # Publishers tracción
        self.pub_encoders = self.create_publisher(Float32MultiArray, '/nixito/encoders', 10)
        self.pub_velocidad = self.create_publisher(Float32MultiArray, '/nixito/velocidad', 10)

        # JointState para TF/URDF
        self.pub_joint_states = self.create_publisher(JointState, '/nixito/joint_states', 10)
        self.pub_drive_encoders = self.create_publisher(JointState, '/nixito/drive_encoders', qos_cmds)

        # Subscriber cmd_vel
        self.create_subscription(
            Float32MultiArray,
            '/cmd_vel',
            self._update_tx_values,
            qos_cmds
        )

        self.tx_values = [0, 0]
        self.send_interval = 0.05  # Envío continuo a 20Hz
        self.last_send_time = 0.0

        try:
            self.ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.1)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            self.get_logger().info("Conectado en /dev/ttyUSB0 a 115200 baudios")
        except Exception as e:
            self.get_logger().error(f"Error crítico: No se pudo abrir el puerto serial: {e}")
            raise e

        self.serial_buffer = ""
        self.timer = self.create_timer(0.02, self.timer_callback)

    def timer_callback(self):
        try:
            # Leer serial
            if self.ser.in_waiting > 0:
                raw = self.ser.read(self.ser.in_waiting)
                self.serial_buffer += raw.decode('utf-8', errors='ignore')

            # Procesar líneas completas
            while '\n' in self.serial_buffer:
                line, self.serial_buffer = self.serial_buffer.split('\n', 1)
                line = line.strip()

                if not line:
                    continue

                if line.startswith("DRV:"):
                    self.procesar_drv(line)
                elif line.startswith("ARM:"):
                    self.procesar_arm(line)
                else:
                    self.get_logger().warn(f"Línea desconocida: {line}")

            # Envío continuo de cmd_vel
            now = time.time()
            if now - self.last_send_time >= self.send_interval:
                self._write_tx_values()
                self.last_send_time = now

        except Exception as e:
            self.get_logger().error(f"Error procesando serial: {e}")

    def _update_tx_values(self, msg):
        data = list(msg.data)

        if len(data) < 2:
            self.get_logger().warn(f"/cmd_vel inválido: {data}")
            return

        try:
            new_values = [int(data[0]), int(data[1])]
            if new_values != self.tx_values:
                self.tx_values = new_values
                self.get_logger().info(f"Nuevo cmd_vel: {self.tx_values}")
        except Exception as e:
            self.get_logger().warn(f"Error convirtiendo /cmd_vel {data}: {e}")

    def _write_tx_values(self):
        payload = f"{self.tx_values[0]},{self.tx_values[1]}\n"
        try:
            self.ser.write(payload.encode('utf-8'))
        except Exception as e:
            self.get_logger().error(f"Error enviando serial '{payload.strip()}': {e}")

    def procesar_drv(self, line):
        try:
            datos = line.replace("DRV:", "").split(',')

            if len(datos) != 4:
                self.get_logger().warn(f"DRV inválido: {datos}")
                return

            values = [float(val.strip()) for val in datos]
            encoders = values[:2]
            velocidad = values[2:]

            self.get_logger().info(f"ENC={encoders} VEL={velocidad}")

            msg_enc = Float32MultiArray()
            msg_enc.data = encoders
            self.pub_encoders.publish(msg_enc)

            msg_vel = Float32MultiArray()
            msg_vel.data = velocidad
            self.pub_velocidad.publish(msg_vel)

            msg_js = JointState()
            msg_js.header.stamp = self.get_clock().now().to_msg()
            msg_js.name = ['dir_R_joint', 'dir_L_joint']
            msg_js.position = [encoders[0], encoders[1]]
            msg_js.velocity = [velocidad[0], velocidad[1]]
            msg_js.effort = []
            self.pub_joint_states.publish(msg_js)

            msg_drive = JointState()
            msg_drive.header.stamp = self.get_clock().now().to_msg()
            msg_drive.name = ['drive_left', 'drive_right']
            msg_drive.position = [encoders[0], encoders[1]]
            msg_drive.velocity = [velocidad[0], velocidad[1]]
            msg_drive.effort = []
            self.pub_drive_encoders.publish(msg_drive)

        except ValueError as e:
            self.get_logger().warn(f"DRV valor no numérico: {e}")
        except Exception as e:
            self.get_logger().warn(f"Error en procesar_drv: {e}")

    def procesar_arm(self, line):
        try:
            parts = line.replace("ARM:", "").split(',')

            if len(parts) != 6:
                self.get_logger().warn(f"ARM inválido: {parts}")
                return

            raw_values = [int(val.strip()) for val in parts]
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
            self.get_logger().warn(f"ARM valor no numérico: {e}")
        except Exception as e:
            self.get_logger().warn(f"Error en procesar_arm: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ControlTest()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Deteniendo nodo por teclado...")
    finally:
        if hasattr(node, 'ser') and node.ser.is_open:
            node.ser.write(b"0,0\n")
            node.ser.close()
            node.get_logger().info("Puerto serial cerrado.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
