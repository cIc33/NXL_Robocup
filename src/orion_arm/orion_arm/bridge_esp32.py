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
        
        # Conexión USB Directa
        self.ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.05)
        
        # Publicador (Lo que lee del ESP32)
        self.sensor_pub = self.create_publisher(Float32MultiArray, 'brazo/angulos_sensores', qos_sensors)
        
        # Suscriptor (Lo que manda al ESP32)
        self.cmd_sub = self.create_subscription(Int8MultiArray, '/brazo/joint_cmds', self.cmd_callback, qos_cmds)
        
        # Servicio de calibración
        self.srv = self.create_service(Trigger, 'brazo/calibrar_zero', self.calibrar_cb)

        # Timer para leer el USB constantemente
        self.timer = self.create_timer(0.01, self.read_serial)
        self.get_logger().info("Puente Serial iniciado. Adiós micro-ROS.")

    def cmd_callback(self, msg):
        # Transforma el arreglo de ROS 2 en texto puro: "CMD e1 e2 e3 e4 seq\n"
        d = msg.data
        comando_str = f"JOINT_CMD {d[1]} {d[2]} {d[3]} {d[4]} {d[0]}\n"
        self.ser.write(comando_str.encode('utf-8'))

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
                    msg = Float32MultiArray()
                    msg.data = [float(partes[1]), float(partes[2]), float(partes[3]), float(partes[4])]
                    self.sensor_pub.publish(msg)
            except Exception as e:
                pass # Ignorar basura serial temporal

def main(args=None):
    rclpy.init(args=args)
    nodo = SerialBridge()
    rclpy.spin(nodo)
    nodo.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()