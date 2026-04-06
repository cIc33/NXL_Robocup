import rclpy
from rclpy.node import Node
import serial
from std_msgs.msg import Int32  

class MQ2(Node):
    def __init__(self):
        super().__init__('mq2_node')
        self.pub_ = self.create_publisher(Int32, '/ADC', 10)

        try:
            self.ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=0.01)
            self.get_logger().info('Serial connection established')
        except serial.SerialException as e:
            self.get_logger().error(f'Error opening serial port: {e}')
            return

        self.timer = self.create_timer(0.02, self.timer_callback)  

    def timer_callback(self):
        while self.ser.in_waiting > 0:
            line = self.ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue

            parts = line.split(',')
            if len(parts) < 2:
                continue

            try:
                raw_value = float(parts[0])  
  
            except ValueError as e:
                self.get_logger().warn(f'Dato inválido: "{line}"')
                continue

            msg = Int32()
            msg.data = int(raw_value)
            self.pub_.publish(msg)
        
def main(args=None):
    rclpy.init(args=args)
    mq2_node = MQ2()
    rclpy.spin(mq2_node)
    mq2_node.destroy_node()
    rclpy.shutdown()