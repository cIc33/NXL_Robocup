#!/usr/bin/env python3
from copy import deepcopy

import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32MultiArray, Int32


class PiperHardwareVelocityBridge(Node):
    ARM_JOINTS = [f'joint{i}' for i in range(1, 7)]
    DRIVER_JOINT_NAMES = ARM_JOINTS + ['gripper']

    def __init__(self):
        super().__init__('piper_hardware_velocity_bridge')

        self.declare_parameter('command_topic', '/piper/test_velocity_cmd')
        self.declare_parameter('switch_mode_topic', '/piper/switch_mode')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('joint_command_topic', '/piper/joint_ctrl_cmd')
        self.declare_parameter('enable_topic', '/piper/enable')
        self.declare_parameter('auto_enable_on_start', False)
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('command_timeout', 0.3)
        self.declare_parameter('arm_max_velocity', [0.35, 0.35, 0.35, 0.35, 0.35, 0.35])
        self.declare_parameter('arm_lower_limits', [-2.618, 0.0, -2.967, -1.745, -1.22, -2.0944])
        self.declare_parameter('arm_upper_limits', [2.618, 3.14, 0.0, 1.745, 1.22, 2.0944])
        self.declare_parameter('gripper_max_velocity', 0.015)
        self.declare_parameter('gripper_lower_limit', 0.0)
        self.declare_parameter('gripper_upper_limit', 0.035)
        self.declare_parameter('arm_command_epsilon', 0.001)
        self.declare_parameter('gripper_command_epsilon', 0.001)
        self.declare_parameter('driver_speed_percent', 20.0)
        self.declare_parameter('gripper_effort', 1.0)
        self.declare_parameter('send_zero_on_timeout', False)

        self.command_topic = str(self.get_parameter('command_topic').value)
        self.switch_mode_topic = str(self.get_parameter('switch_mode_topic').value)
        self.joint_states_topic = str(self.get_parameter('joint_states_topic').value)
        self.joint_command_topic = str(self.get_parameter('joint_command_topic').value)
        self.enable_topic = str(self.get_parameter('enable_topic').value)
        self.auto_enable_on_start = bool(self.get_parameter('auto_enable_on_start').value)
        self.publish_rate = float(self.get_parameter('publish_rate').value)
        self.command_timeout = float(self.get_parameter('command_timeout').value)
        self.arm_max_velocity = [float(v) for v in self.get_parameter('arm_max_velocity').value]
        self.arm_lower_limits = [float(v) for v in self.get_parameter('arm_lower_limits').value]
        self.arm_upper_limits = [float(v) for v in self.get_parameter('arm_upper_limits').value]
        self.gripper_max_velocity = float(self.get_parameter('gripper_max_velocity').value)
        self.gripper_lower_limit = float(self.get_parameter('gripper_lower_limit').value)
        self.gripper_upper_limit = float(self.get_parameter('gripper_upper_limit').value)
        self.arm_command_epsilon = float(self.get_parameter('arm_command_epsilon').value)
        self.gripper_command_epsilon = float(self.get_parameter('gripper_command_epsilon').value)
        self.driver_speed_percent = float(self.get_parameter('driver_speed_percent').value)
        self.gripper_effort = float(self.get_parameter('gripper_effort').value)
        self.send_zero_on_timeout = bool(self.get_parameter('send_zero_on_timeout').value)

        self.create_subscription(Float32MultiArray, self.command_topic, self.command_callback, 10)
        self.create_subscription(Int32, self.switch_mode_topic, self.switch_mode_callback, 10)
        self.create_subscription(JointState, self.joint_states_topic, self.joint_state_callback, 10)
        self.command_publisher = self.create_publisher(JointState, self.joint_command_topic, 10)
        self.enable_publisher = self.create_publisher(Bool, self.enable_topic, 10)

        self.current_state = {}
        self.current_stamp = None
        self.command_values = [0.0] * 7
        self.command_stamp = None
        self.cartesian_mode_requested = False
        self.arm_target_state = None
        self.gripper_target_state = None
        self.warned_cartesian = False

        if self.auto_enable_on_start:
            enable = Bool()
            enable.data = True
            self.enable_publisher.publish(enable)

        self.create_timer(1.0 / self.publish_rate, self.timer_callback)
        self.get_logger().info(
            f'Hardware velocity bridge ready. Commands: {self.command_topic}, output: {self.joint_command_topic}. '
            'Mode 0=joint hardware bridge, mode 1 is ignored here.'
        )

    def command_callback(self, msg: Float32MultiArray):
        if len(msg.data) != 7:
            self.get_logger().warn('Ignoring command: expected exactly 7 values')
            return
        values = [float(v) for v in msg.data]
        for index in range(6):
            values[index] = max(-100.0, min(100.0, values[index]))
        values[6] = max(-1.0, min(1.0, values[6]))
        self.command_values = values
        self.command_stamp = self.get_clock().now()

    def switch_mode_callback(self, msg: Int32):
        self.cartesian_mode_requested = msg.data == 1
        if not self.cartesian_mode_requested:
            self.warned_cartesian = False

    def joint_state_callback(self, msg: JointState):
        self.current_state = {
            name: msg.position[index]
            for index, name in enumerate(msg.name)
            if index < len(msg.position)
        }
        self.current_stamp = self.get_clock().now()
        if self.arm_target_state is None:
            self.arm_target_state = [self.current_state.get(name, 0.0) for name in self.ARM_JOINTS]
        if self.gripper_target_state is None:
            self.gripper_target_state = self.get_current_gripper_position()

    def timer_callback(self):
        if self.current_stamp is None:
            return

        if self.cartesian_mode_requested:
            if not self.warned_cartesian:
                self.get_logger().warn('Ignoring mode 1: hardware_velocity_bridge handles joint mode only')
                self.warned_cartesian = True
            return

        now = self.get_clock().now()
        timed_out = self.command_stamp is None or (now - self.command_stamp) > Duration(seconds=self.command_timeout)
        if timed_out:
            command = [0.0] * 7
        else:
            command = deepcopy(self.command_values)

        arm_active = any(abs(value) > self.arm_command_epsilon for value in command[:6])
        gripper_active = abs(command[6]) > self.gripper_command_epsilon
        if not arm_active and not gripper_active and not self.send_zero_on_timeout:
            self.arm_target_state = [self.current_state.get(name, 0.0) for name in self.ARM_JOINTS]
            self.gripper_target_state = self.get_current_gripper_position()
            return

        if self.arm_target_state is None:
            self.arm_target_state = [self.current_state.get(name, 0.0) for name in self.ARM_JOINTS]
        if self.gripper_target_state is None:
            self.gripper_target_state = self.get_current_gripper_position()

        dt = 1.0 / self.publish_rate
        if arm_active:
            arm_targets = []
            for index in range(6):
                velocity = (command[index] / 100.0) * self.arm_max_velocity[index]
                target = self.arm_target_state[index] + velocity * dt
                target = max(self.arm_lower_limits[index], min(self.arm_upper_limits[index], target))
                arm_targets.append(target)
            self.arm_target_state = arm_targets

        if gripper_active:
            target = self.gripper_target_state + command[6] * self.gripper_max_velocity * dt
            self.gripper_target_state = max(self.gripper_lower_limit, min(self.gripper_upper_limit, target))

        self.publish_joint_command(self.arm_target_state, self.gripper_target_state)

    def get_current_gripper_position(self):
        if 'gripper' in self.current_state:
            return self.current_state['gripper']
        if 'joint7' in self.current_state:
            return max(0.0, self.current_state['joint7'])
        return 0.0

    def publish_joint_command(self, arm_positions, gripper_position):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.DRIVER_JOINT_NAMES
        msg.position = list(arm_positions) + [float(gripper_position)]
        msg.velocity = [0.0] * 6 + [max(1.0, min(100.0, self.driver_speed_percent))]
        msg.effort = [0.0] * 6 + [max(0.5, min(3.0, self.gripper_effort))]
        self.command_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PiperHardwareVelocityBridge()
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
