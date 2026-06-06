#!/usr/bin/env python3
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState


class PiperJointStateNormalizer(Node):
    ARM_JOINTS = [f'joint{i}' for i in range(1, 7)]

    def __init__(self):
        super().__init__('piper_joint_state_normalizer')

        self.declare_parameter('input_topic', '/piper/joint_states_feedback')
        self.declare_parameter('output_topic', '/joint_states')
        self.declare_parameter('gripper_input_name', 'gripper')
        self.declare_parameter('include_gripper', True)

        self.input_topic = str(self.get_parameter('input_topic').value)
        self.output_topic = str(self.get_parameter('output_topic').value)
        self.gripper_input_name = str(self.get_parameter('gripper_input_name').value)
        self.include_gripper = bool(self.get_parameter('include_gripper').value)

        self.publisher = self.create_publisher(JointState, self.output_topic, 10)
        self.create_subscription(JointState, self.input_topic, self.joint_state_callback, 10)

        self.get_logger().info(
            f'Normalizing Piper feedback from {self.input_topic} to {self.output_topic}'
        )

    def joint_state_callback(self, msg: JointState):
        positions = {name: msg.position[index] for index, name in enumerate(msg.name) if index < len(msg.position)}
        velocities = {name: msg.velocity[index] for index, name in enumerate(msg.name) if index < len(msg.velocity)}
        efforts = {name: msg.effort[index] for index, name in enumerate(msg.name) if index < len(msg.effort)}

        output = JointState()
        output.header = msg.header
        output.name = list(self.ARM_JOINTS)
        output.position = [positions.get(name, 0.0) for name in self.ARM_JOINTS]
        output.velocity = [velocities.get(name, 0.0) for name in self.ARM_JOINTS]
        output.effort = [efforts.get(name, 0.0) for name in self.ARM_JOINTS]

        if self.include_gripper:
            gripper = positions.get(self.gripper_input_name, positions.get('joint7', 0.0))
            gripper_velocity = velocities.get(self.gripper_input_name, velocities.get('joint7', 0.0))
            gripper_effort = efforts.get(self.gripper_input_name, efforts.get('joint7', 0.0))
            output.name.extend(['joint7', 'joint8'])
            output.position.extend([max(0.0, gripper), -max(0.0, gripper)])
            output.velocity.extend([gripper_velocity, -gripper_velocity])
            output.effort.extend([gripper_effort, -gripper_effort])

        self.publisher.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = PiperJointStateNormalizer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
