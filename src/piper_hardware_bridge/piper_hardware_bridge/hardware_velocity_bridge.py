#!/usr/bin/env python3
import math
import time
from copy import deepcopy

import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from geometry_msgs.msg import Pose
from piper_msgs.msg import PosCmd
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
        self.declare_parameter('gripper_max_velocity', 0.05)
        self.declare_parameter('gripper_lower_limit', 0.0)
        self.declare_parameter('gripper_upper_limit', 0.1)
        self.declare_parameter('arm_command_epsilon', 0.001)
        self.declare_parameter('gripper_command_epsilon', 0.001)
        self.declare_parameter('min_publish_delta', 0.003)
        self.declare_parameter('driver_speed_percent', 60.0)
        self.declare_parameter('gripper_effort', 1.0)
        self.declare_parameter('send_zero_on_timeout', False)
        self.declare_parameter('gripper_keepalive_hz', 5.0)
        self.declare_parameter('cartesian_pos_cmd_topic', '/piper/pos_cmd')
        self.declare_parameter('end_pose_topic', '/piper/end_pose')
        self.declare_parameter('cartesian_max_linear_velocity', 0.05)
        self.declare_parameter('cartesian_max_angular_velocity', 0.3)

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
        self.min_publish_delta = float(self.get_parameter('min_publish_delta').value)
        gripper_keepalive_hz = float(self.get_parameter('gripper_keepalive_hz').value)
        self.gripper_keepalive_interval = 1.0 / gripper_keepalive_hz if gripper_keepalive_hz > 0 else 0.0
        self.cartesian_pos_cmd_topic = str(self.get_parameter('cartesian_pos_cmd_topic').value)
        self.end_pose_topic = str(self.get_parameter('end_pose_topic').value)
        self.cartesian_max_linear_velocity = float(self.get_parameter('cartesian_max_linear_velocity').value)
        self.cartesian_max_angular_velocity = float(self.get_parameter('cartesian_max_angular_velocity').value)

        self.last_published_arm = None
        self.last_published_gripper = None
        self.last_keepalive_time = 0.0
        self.last_arm_move_time = 0.0
        self.cartesian_target = None  # [x, y, z, roll, pitch, yaw] in m / rad

        self.create_subscription(Float32MultiArray, self.command_topic, self.command_callback, 10)
        self.create_subscription(Int32, self.switch_mode_topic, self.switch_mode_callback, 10)
        self.create_subscription(JointState, self.joint_states_topic, self.joint_state_callback, 10)
        self.create_subscription(Pose, self.end_pose_topic, self.end_pose_callback, 10)
        self.command_publisher = self.create_publisher(JointState, self.joint_command_topic, 10)
        self.pos_cmd_publisher = self.create_publisher(PosCmd, self.cartesian_pos_cmd_topic, 10)
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
        entering_cartesian = msg.data == 1 and not self.cartesian_mode_requested
        self.cartesian_mode_requested = msg.data == 1
        if entering_cartesian:
            self.cartesian_target = None  # re-initialize from current pose on next tick
        if not self.cartesian_mode_requested:
            self.warned_cartesian = False

    def end_pose_callback(self, msg: Pose):
        if self.cartesian_target is None:
            roll, pitch, yaw = self._quat_to_euler(
                msg.orientation.x, msg.orientation.y,
                msg.orientation.z, msg.orientation.w,
            )
            self.cartesian_target = [
                msg.position.x, msg.position.y, msg.position.z,
                roll, pitch, yaw,
            ]

    def joint_state_callback(self, msg: JointState):
        new_state = {
            name: msg.position[index]
            for index, name in enumerate(msg.name)
            if index < len(msg.position)
        }
        # Detect intentional arm movement (> 0.05 rad/s at 200 Hz = 0.00025 rad/tick).
        # Below this threshold is encoder noise; above is any real trajectory or GUI command.
        if self.current_state:
            arm_moved = any(
                abs(new_state.get(n, 0.0) - self.current_state.get(n, 0.0)) > 0.00025
                for n in self.ARM_JOINTS
            )
            if arm_moved:
                self.last_arm_move_time = time.monotonic()
        self.current_state = new_state
        self.current_stamp = self.get_clock().now()
        if self.arm_target_state is None:
            self.arm_target_state = [self.current_state.get(name, 0.0) for name in self.ARM_JOINTS]
        if self.gripper_target_state is None:
            self.gripper_target_state = self.get_current_gripper_position()

    def timer_callback(self):
        if self.current_stamp is None:
            return

        now = self.get_clock().now()
        timed_out = self.command_stamp is None or (now - self.command_stamp) > Duration(seconds=self.command_timeout)
        if timed_out:
            command = [0.0] * 7
        else:
            command = deepcopy(self.command_values)

        dt = 1.0 / self.publish_rate

        if self.cartesian_mode_requested:
            self._handle_cartesian(command, dt)
            return

        arm_active = any(abs(value) > self.arm_command_epsilon for value in command[:6])
        gripper_active = abs(command[6]) > self.gripper_command_epsilon
        if not arm_active and not gripper_active and not self.send_zero_on_timeout:
            self.arm_target_state = [self.current_state.get(name, 0.0) for name in self.ARM_JOINTS]
            # gripper_target_state is NOT updated from feedback here — it tracks the last
            # commanded value. Reading the normalized feedback (÷2) and re-sending it as a
            # command would halve the gripper position on every keepalive tick.
            now = time.monotonic()
            if self.gripper_target_state is not None and (now - self.last_keepalive_time) >= self.gripper_keepalive_interval:
                self.last_keepalive_time = now
                self._publish_gripper_keepalive()
            return

        if self.arm_target_state is None:
            self.arm_target_state = [self.current_state.get(name, 0.0) for name in self.ARM_JOINTS]
        if self.gripper_target_state is None:
            self.gripper_target_state = self.get_current_gripper_position()

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

    def _handle_cartesian(self, command, dt):
        if self.cartesian_target is None:
            return  # waiting for first end_pose feedback

        linear_active = any(abs(command[i]) > self.arm_command_epsilon for i in range(3))
        angular_active = any(abs(command[i + 3]) > self.arm_command_epsilon for i in range(3))
        gripper_active = abs(command[6]) > self.gripper_command_epsilon

        if not linear_active and not angular_active and not gripper_active:
            return  # arm holds position in SDK; gripper keepalive handled separately

        if linear_active or angular_active:
            for i in range(3):
                self.cartesian_target[i] += (command[i] / 100.0) * self.cartesian_max_linear_velocity * dt
            for i in range(3):
                self.cartesian_target[i + 3] += (command[i + 3] / 100.0) * self.cartesian_max_angular_velocity * dt

        if gripper_active:
            target = self.gripper_target_state + command[6] * self.gripper_max_velocity * dt
            self.gripper_target_state = max(self.gripper_lower_limit, min(self.gripper_upper_limit, target))

        msg = PosCmd()
        msg.x, msg.y, msg.z = self.cartesian_target[0], self.cartesian_target[1], self.cartesian_target[2]
        msg.roll, msg.pitch, msg.yaw = self.cartesian_target[3], self.cartesian_target[4], self.cartesian_target[5]
        msg.gripper = float(self.gripper_target_state) if self.gripper_target_state is not None else 0.0
        msg.mode1 = 0
        msg.mode2 = 0
        self.pos_cmd_publisher.publish(msg)

    @staticmethod
    def _quat_to_euler(x, y, z, w):
        sinr = 2.0 * (w * x + y * z)
        cosr = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr, cosr)
        sinp = 2.0 * (w * y - z * x)
        pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
        siny = 2.0 * (w * z + x * y)
        cosy = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny, cosy)
        return roll, pitch, yaw

    def _publish_gripper_keepalive(self):
        if self.cartesian_mode_requested:
            return  # cartesian mode uses pos_cmd; don't switch SDK to joint mode
        # Suppress during trajectory execution — arm feedback changes trigger last_arm_move_time.
        if time.monotonic() - self.last_arm_move_time < 2.0:
            return
        arm_positions = [self.current_state.get(name, 0.0) for name in self.ARM_JOINTS]
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.DRIVER_JOINT_NAMES
        msg.position = list(arm_positions) + [float(self.gripper_target_state)]
        msg.velocity = [0.0] * 6 + [max(1.0, min(100.0, self.driver_speed_percent))]
        msg.effort = [0.0] * 6 + [max(0.5, min(3.0, self.gripper_effort))]
        self.command_publisher.publish(msg)

    def get_current_gripper_position(self):
        if 'gripper' in self.current_state:
            return self.current_state['gripper']
        if 'joint7' in self.current_state:
            return max(0.0, self.current_state['joint7'])
        return 0.0

    def publish_joint_command(self, arm_positions, gripper_position):
        arm_changed = self.last_published_arm is None or any(
            abs(a - b) >= self.min_publish_delta
            for a, b in zip(arm_positions, self.last_published_arm)
        )
        gripper_changed = self.last_published_gripper is None or (
            abs(gripper_position - self.last_published_gripper) >= self.min_publish_delta
        )
        if not arm_changed and not gripper_changed:
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.DRIVER_JOINT_NAMES
        msg.position = list(arm_positions) + [float(gripper_position)]
        msg.velocity = [0.0] * 6 + [max(1.0, min(100.0, self.driver_speed_percent))]
        msg.effort = [0.0] * 6 + [max(0.5, min(3.0, self.gripper_effort))]
        self.command_publisher.publish(msg)
        self.last_published_arm = list(arm_positions)
        self.last_published_gripper = float(gripper_position)


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
