import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
from rclpy.qos import qos_profile_sensor_data
import math

class RealStatePublisher(Node):
    def __init__(self):
        super().__init__('real_state_publisher')

        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/brazo/angulos_sensores',
            self.listener_callback,
            qos_profile_sensor_data
        )

        # IMPORTANTE: absoluto
        self.publisher_ = self.create_publisher(JointState, '/nixito/joint_states', qos_profile_sensor_data)

        self.joint_names = [
            'Hombro_joint',
            'E1_joint',
            'E2_joint',
            'E3_joint'
        ]

        self.get_logger().info('Nodo Bridge brazo iniciado. Esperando datos...')

    def listener_callback(self, msg):
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = self.joint_names

        try:
            radianes = [math.radians(val) for val in msg.data]

            while len(radianes) < 4:
                radianes.append(0.0)

            joint_state.position = radianes[:4]
            self.publisher_.publish(joint_state)

        except Exception as e:
            self.get_logger().error(f"Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = RealStatePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()