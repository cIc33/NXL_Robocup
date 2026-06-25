import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import serial
import time
import math 

# ==============================
# FUNCIONES AUXILIARES MODBUS RTU
# ==============================
def modbus_crc(data: bytes) -> bytes:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc >>= 1
                crc ^= 0xA001
            else:
                crc >>= 1
    return bytes([crc & 0xFF, (crc >> 8) & 0xFF])

def check_crc(frame: bytes) -> bool:
    if len(frame) < 4:
        return False
    return frame[-2:] == modbus_crc(frame[:-2])


class ControlTest(Node):
    def __init__(self):
        # AÑADIDO: Mantenemos el namespace por si tienes aislado el robot_state_publisher de Nixito
        super().__init__('control_test', namespace='nixito')

        qos_cmds = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # === PUBLISHERS ORIGINALES DEL ARDUINO ===
        self.pub_piper = self.create_publisher(Float32MultiArray, '/piper/test_velocity_cmd', qos_cmds)
        self.pub_piper_mode = self.create_publisher(Int32, '/piper/switch_mode', qos_cmds)
        self.pub_flip_encoders = self.create_publisher(Float32MultiArray, '/flipper_encoders', qos_cmds)

        # === PUBLISHERS PARA LOS ENCODERS RS485 ===
        self.pub_abs_encoders = self.create_publisher(Float32MultiArray, '/abs_encoders', qos_cmds)
        
        # AÑADIDO: Publisher para el URDF/RViz
        self.pub_joint_state = self.create_publisher(JointState, 'joint_states', qos_cmds)

        # === SUBSCRIBER ORIGINAL ===
        self.create_subscription(
            Float32MultiArray,
            '/cmd_vel',
            self._update_tx_values,
            qos_cmds
        )

        self.tx_values = [0, 0]
        self.send_interval = 0.05  # Envío continuo a 20Hz
        self.last_send_time = 0.0

        # === MAPEO DE switch_mode ===
        # El Arduino manda -100 o 100 en el último campo de la trama ARM8.
        # Se traduce a un valor binario simple para /piper/switch_mode
        self.switch_mode_map = {100: 0, -100: 1}
        self._last_switch_mode = 0  # valor por defecto / último válido conocido

        # === CONEXIÓN ORIGINAL: ARDUINO MEGA PRO ===
        # FIX: este puerto estaba duplicado con el de RS485 (ambos en ttyUSB1).
        # Se restaura a ttyUSB0 -- confirma que sea la asignación física correcta.
        try:
            self.ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.1)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            self.get_logger().info("Arduino conectado en /dev/ttyUSB0 a 115200 baudios")
        except Exception as e:
            self.get_logger().error(f"Error crítico: No se pudo abrir el puerto serial: {e}")
            raise e
        
        # === CONEXIÓN RS485 MODBUS ===
        self.ser_rs485 = None
        self.rs485_connected = False  # Bandera para saber si el puerto está listo

        try:
            self.ser_rs485 = serial.Serial('/dev/ttyUSB2', 9600, timeout=0.05)
            self.ser_rs485.reset_input_buffer()
            self.ser_rs485.reset_output_buffer()
            self.rs485_connected = True
            self.get_logger().info("Encoders Modbus conectados en /dev/ttyUSB2 a 9600 baudios")
        except Exception as e:
            # Quitamos el 'raise e' para que el nodo NO muera si no detecta el USB1
            self.get_logger().warn(f"¡Atención! No se pudo abrir el puerto RS485 (/dev/ttyUSB1): {e}. El nodo continuará sin encoders.")
        # Variables y timers
        self.serial_buffer = ""
        self.encoder_ids = [3, 5]
        
        # AÑADIDO: Memoria de radianes para publicar continuamente el JointState incluso si falla una lectura
        self._last_valid_rad = [0.0, 0.0]
        
        # Timer original para Arduino (50Hz)
        self.timer = self.create_timer(0.02, self.timer_callback)
        
        # NUEVO Timer independiente para Modbus (10Hz)
        self.timer_modbus = self.create_timer(0.1, self.timer_modbus_callback)

    # ========================================================
    # LÓGICA ORIGINAL INTACTA: ARDUINO MEGA PRO
    # ========================================================
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

                if line.startswith("DRV:"):
                    self.procesar_drv(line)
                elif line.startswith("ARM8:"):
                    self.procesar_piper(line)
                else:
                    self.get_logger().warn(f"Línea desconocida: {line}")

            now = time.time()
            if now - self.last_send_time >= self.send_interval:
                self._write_tx_values()
                self.last_send_time = now

        except Exception as e:
            self.get_logger().error(f"Error procesando serial Arduino: {e}")

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
        pass

    def procesar_piper(self, line):
        try:
            parts = line.replace("ARM8:", "").split(',')

            if len(parts) != 8:
                self.get_logger().warn(f"ARM inválido: {parts}")
                return

            raw_values = [int(val.strip()) for val in parts]

            # Brazo y gripper se publican TAL CUAL llegan, sin escalar (rango -100 a 100)
            brazo = raw_values[:7]

            # === MAPEO switch_mode: 100 -> 0, -100 -> 1 ===
            raw_switch = raw_values[7]
            if raw_switch in self.switch_mode_map:
                switch_mode = self.switch_mode_map[raw_switch]
                self._last_switch_mode = switch_mode
            else:
                self.get_logger().warn(
                    f"switch_mode inesperado: {raw_switch} (se esperaba 100 o -100); "
                    f"se mantiene último valor válido: {self._last_switch_mode}"
                )
                switch_mode = self._last_switch_mode

            msg_brazo = Float32MultiArray()
            msg_brazo.data = [float(v) for v in brazo]
            self.pub_piper.publish(msg_brazo)

            msg_mode = Int32()
            msg_mode.data = switch_mode
            self.pub_piper_mode.publish(msg_mode)
            
            msg_encoders = Float32MultiArray()
            msg_encoders.data = [float(x) for x in raw_values]
            self.pub_flip_encoders.publish(msg_encoders)

        except ValueError as e:
            self.get_logger().warn(f"PIPER valor no numérico: {e}")
        except Exception as e:
            self.get_logger().warn(f"Error en procesar_piper: {e}")

    # ========================================================
    # LÓGICA AISLADA: ENCODERS ABSOLUTOS MODBUS
    # ========================================================
    def timer_modbus_callback(self):
        # Si el puerto serial no se pudo abrir al inicio, saltamos la lectura Modbus
        # pero dejamos que siga abajo para publicar el último JointState válido.
        if self.rs485_connected and self.ser_rs485 is not None:
            angles = []
            
            for enc_id in self.encoder_ids:
                pos, angle, status = self.interrogar_encoder(enc_id)
                
                if status == "OK":
                    angles.append(angle)
                else:
                    self.get_logger().warn(f"RS485 ID {enc_id}: Error {status}")
                    angles.append(-1.0) 
                
                time.sleep(0.01)
                
            if len(angles) == 2 and angles[0] != -1.0 and angles[1] != -1.0:
                # ==========================================
                # APLICACIÓN DE OFFSETS (Cero Absoluto)
                # ==========================================
                angles[0] = 360.0 - angles[0]
                offset_enc1 = -215.5078125  
                offset_enc2 = 109.3359375  
                
                angulo_corregido_1 = (angles[0] - offset_enc1) % 360.0
                angulo_corregido_2 = (angles[1] - offset_enc2) % 360.0

                msg = Float32MultiArray()
                msg.data = [angulo_corregido_1, angulo_corregido_2]
                self.pub_abs_encoders.publish(msg)

                # Convertir a radianes para RViz
                rad_R = math.radians(angulo_corregido_1)
                rad_L = math.radians(angulo_corregido_2)

                rad_R = (rad_R + math.pi) % (2 * math.pi) - math.pi
                rad_L = (rad_L + math.pi) % (2 * math.pi) - math.pi

                self._last_valid_rad = [rad_R, rad_L]

        # Esto se ejecuta SIEMPRE. Si no hay USB1, publicará [0.0, 0.0] (o los últimos radianes que haya guardado)
        self._publicar_joint_state()

    def _publicar_joint_state(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        # Nombres exactos de las articulaciones de los flippers en tu modelo
        msg.name = ['flipper_R_joint', 'flipper_L_joint']
        msg.position = self._last_valid_rad   
        msg.velocity = []   
        msg.effort = []

        self.pub_joint_state.publish(msg)

    def interrogar_encoder(self, encoder_id):
        request = bytes([encoder_id, 0x03, 0x00, 0x00, 0x00, 0x01])
        frame = request + modbus_crc(request)

        try:
            self.ser_rs485.reset_input_buffer()
            self.ser_rs485.write(frame)
            self.ser_rs485.flush()

            response = self.ser_rs485.read(7)

            if len(response) < 7:
                return None, None, "TIMEOUT"
            if not check_crc(response):
                return None, None, "CRC_ERROR"
            if response[0] != encoder_id:
                return None, None, "WRONG_ID"

            position = (response[3] << 8) | response[4]
            angle = position * 360.0 / 1024.0 
            return position, angle, "OK"

        except Exception as e:
            return None, None, f"SERIAL_FAULT: {e}"


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
        # Solo cerramos el RS485 si la bandera fue True y el objeto existe
        if hasattr(node, 'ser_rs485') and node.ser_rs485 is not None and node.ser_rs485.is_open:
            node.ser_rs485.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()