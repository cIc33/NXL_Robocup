#!/usr/bin/env python3
import math
from copy import deepcopy

import rclpy
import tf2_ros
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, Quaternion
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.srv import GetCartesianPath
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, Int32
from tf2_ros import TransformException
from trajectory_msgs.msg import JointTrajectoryPoint


def _euler_to_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


def _quat_mul(q1, q2):
    q = Quaternion()
    q.w = q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z
    q.x = q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y
    q.y = q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x
    q.z = q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w
    return q


def _rotate_vec(v, q):
    u = (q.x, q.y, q.z)
    s = q.w
    dot = u[0]*v[0] + u[1]*v[1] + u[2]*v[2]
    uu = u[0]*u[0] + u[1]*u[1] + u[2]*u[2]
    cx = u[1]*v[2] - u[2]*v[1]
    cy = u[2]*v[0] - u[0]*v[2]
    cz = u[0]*v[1] - u[1]*v[0]
    return (
        2*dot*u[0] + (s*s - uu)*v[0] + 2*s*cx,
        2*dot*u[1] + (s*s - uu)*v[1] + 2*s*cy,
        2*dot*u[2] + (s*s - uu)*v[2] + 2*s*cz,
    )


class PiperVelocityTeleop(Node):
    ARM_JOINTS = [f'joint{i}' for i in range(1, 7)]
    GRIPPER_JOINT = 'joint7'

    def __init__(self):
        super().__init__('piper_velocity_teleop')

        self.declare_parameter('command_topic', '/piper/test_velocity_cmd')
        self.declare_parameter('switch_mode_topic', '/piper/switch_mode')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('command_timeout', 0.3)
        self.declare_parameter('arm_action_name', '/arm_controller/follow_joint_trajectory')
        self.declare_parameter('gripper_action_name', '/gripper_controller/follow_joint_trajectory')
        self.declare_parameter('arm_max_velocity', [0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
        self.declare_parameter('arm_lower_limits', [-2.618, 0.0, -2.967, -1.745, -1.22, -2.0944])
        self.declare_parameter('arm_upper_limits', [2.618, 3.14, 0.0, 1.745, 1.22, 2.0944])
        self.declare_parameter('gripper_max_velocity', 0.03)
        self.declare_parameter('gripper_lower_limit', 0.0)
        self.declare_parameter('gripper_upper_limit', 0.035)
        self.declare_parameter('trajectory_duration', 0.15)
        self.declare_parameter('arm_command_epsilon', 0.001)
        self.declare_parameter('gripper_command_epsilon', 0.001)
        # Cartesian MoveGroup parameters
        self.declare_parameter('planning_frame', 'base_link')
        self.declare_parameter('ee_link', 'link6')
        self.declare_parameter('arm_group_name', 'arm')
        self.declare_parameter('cartesian_linear_scale', 0.0004)
        self.declare_parameter('cartesian_rotational_scale', 0.002)
        self.declare_parameter('cartesian_max_step', 0.005)
        self.declare_parameter('cartesian_fraction_threshold', 0.5)

        self.command_topic = str(self.get_parameter('command_topic').value)
        self.switch_mode_topic = str(self.get_parameter('switch_mode_topic').value)
        self.joint_states_topic = str(self.get_parameter('joint_states_topic').value)
        self.publish_rate = float(self.get_parameter('publish_rate').value)
        self.command_timeout = float(self.get_parameter('command_timeout').value)
        self.arm_action_name = str(self.get_parameter('arm_action_name').value)
        self.gripper_action_name = str(self.get_parameter('gripper_action_name').value)
        self.arm_max_velocity = [float(v) for v in self.get_parameter('arm_max_velocity').value]
        self.arm_lower_limits = [float(v) for v in self.get_parameter('arm_lower_limits').value]
        self.arm_upper_limits = [float(v) for v in self.get_parameter('arm_upper_limits').value]
        self.gripper_max_velocity = float(self.get_parameter('gripper_max_velocity').value)
        self.gripper_lower_limit = float(self.get_parameter('gripper_lower_limit').value)
        self.gripper_upper_limit = float(self.get_parameter('gripper_upper_limit').value)
        self.trajectory_duration = float(self.get_parameter('trajectory_duration').value)
        self.arm_command_epsilon = float(self.get_parameter('arm_command_epsilon').value)
        self.gripper_command_epsilon = float(self.get_parameter('gripper_command_epsilon').value)
        self.planning_frame = str(self.get_parameter('planning_frame').value)
        self.ee_link = str(self.get_parameter('ee_link').value)
        self.arm_group_name = str(self.get_parameter('arm_group_name').value)
        self.cartesian_linear_scale = float(self.get_parameter('cartesian_linear_scale').value)
        self.cartesian_rotational_scale = float(self.get_parameter('cartesian_rotational_scale').value)
        self.cartesian_max_step = float(self.get_parameter('cartesian_max_step').value)
        self.cartesian_fraction_threshold = float(self.get_parameter('cartesian_fraction_threshold').value)

        self.create_subscription(Float32MultiArray, self.command_topic, self.command_callback, 10)
        self.create_subscription(Int32, self.switch_mode_topic, self.switch_mode_callback, 10)
        self.create_subscription(JointState, self.joint_states_topic, self.joint_state_callback, 10)

        self.arm_action = ActionClient(self, FollowJointTrajectory, self.arm_action_name)
        self.gripper_action = ActionClient(self, FollowJointTrajectory, self.gripper_action_name)

        # MoveGroup Cartesian
        self.cartesian_path_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        self.execute_action = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.current_state = {}
        self.current_stamp = None
        self.command_values = [0.0] * 7
        self.command_stamp = None
        self.cartesian_mode_requested = False
        self.cartesian_executing = False
        self.arm_target_state = None
        self.gripper_target_state = None
        self.last_arm_action_send_time = None
        self.last_gripper_action_send_time = None
        self.last_gripper_goal_position = None

        self.create_timer(1.0 / self.publish_rate, self.timer_callback)
        self.get_logger().info(
            f'Piper teleop ready. Mode 0=joint, mode 1=cartesian (MoveGroup). '
            f'linear_scale={self.cartesian_linear_scale}, rot_scale={self.cartesian_rotational_scale}'
        )

    # ── Subscriptions ─────────────────────────────────────────────────────────

    def command_callback(self, msg: Float32MultiArray):
        if len(msg.data) != 7:
            return
        values = [float(v) for v in msg.data]
        for i in range(6):
            values[i] = max(-100.0, min(100.0, values[i]))
        values[6] = max(-1.0, min(1.0, values[6]))
        self.command_values = values
        self.command_stamp = self.get_clock().now()

    def switch_mode_callback(self, msg: Int32):
        previous = self.cartesian_mode_requested
        self.cartesian_mode_requested = msg.data == 1
        if previous != self.cartesian_mode_requested:
            mode = 'cartesian' if self.cartesian_mode_requested else 'joint'
            self.get_logger().info(f'Mode switch → {mode}')
            self.command_values = [0.0] * 7
            self.command_stamp = self.get_clock().now()
            if not self.cartesian_mode_requested:
                # Snapshot current joints so joint mode picks up from here
                self.arm_target_state = [
                    self.current_state.get(name, 0.0) for name in self.ARM_JOINTS
                ]
                self.cartesian_executing = False

    def joint_state_callback(self, msg: JointState):
        self.current_state = {
            name: msg.position[i]
            for i, name in enumerate(msg.name)
            if i < len(msg.position)
        }
        self.current_stamp = self.get_clock().now()
        if self.arm_target_state is None:
            self.arm_target_state = [
                self.current_state.get(name, 0.0) for name in self.ARM_JOINTS
            ]
        if self.gripper_target_state is None:
            self.gripper_target_state = self.current_state.get(self.GRIPPER_JOINT, 0.0)

    # ── Timer ─────────────────────────────────────────────────────────────────

    def timer_callback(self):
        if self.current_stamp is None:
            return

        now = self.get_clock().now()
        if self.command_stamp is None or (now - self.command_stamp) > Duration(seconds=self.command_timeout):
            command = [0.0] * 7
        else:
            command = deepcopy(self.command_values)

        arm_active = any(abs(v) > self.arm_command_epsilon for v in command[:6])
        gripper_active = abs(command[6]) > self.gripper_command_epsilon

        if self.cartesian_mode_requested:
            if arm_active and not self.cartesian_executing:
                self._start_cartesian_step(command[:6])
        else:
            if not arm_active:
                self.arm_target_state = [
                    self.current_state.get(name, 0.0) for name in self.ARM_JOINTS
                ]
                arm_targets = self.arm_target_state
            else:
                dt = 1.0 / self.publish_rate
                arm_targets = []
                for i in range(len(self.ARM_JOINTS)):
                    vel = (command[i] / 100.0) * self.arm_max_velocity[i]
                    target = self.arm_target_state[i] + vel * dt
                    target = max(self.arm_lower_limits[i], min(self.arm_upper_limits[i], target))
                    arm_targets.append(target)
                self.arm_target_state = arm_targets
            if arm_active:
                self.send_arm_action(arm_targets)

        if self.gripper_target_state is None:
            self.gripper_target_state = self.current_state.get(self.GRIPPER_JOINT, 0.0)
        if gripper_active:
            gripper_target = self.gripper_target_state + (
                command[6] * self.gripper_max_velocity / self.publish_rate
            )
            gripper_target = max(self.gripper_lower_limit, min(self.gripper_upper_limit, gripper_target))
            self.gripper_target_state = gripper_target
            self.send_gripper_action(gripper_target)

    # ── Cartesian MoveGroup ───────────────────────────────────────────────────

    def _get_ee_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.planning_frame, self.ee_link, rclpy.time.Time()
            )
            p = Pose()
            p.position.x = tf.transform.translation.x
            p.position.y = tf.transform.translation.y
            p.position.z = tf.transform.translation.z
            p.orientation = tf.transform.rotation
            return p
        except TransformException as e:
            self.get_logger().warn(f'TF lookup failed: {e}', throttle_duration_sec=2.0)
            return None

    def _apply_delta(self, pose: Pose, command) -> Pose:
        target = deepcopy(pose)
        # Linear: commanded in TCP frame, rotate to planning frame
        dx, dy, dz = (command[i] * self.cartesian_linear_scale for i in range(3))
        rx, ry, rz = _rotate_vec((dx, dy, dz), pose.orientation)
        target.position.x += rx
        target.position.y += ry
        target.position.z += rz
        # Angular: applied in TCP frame
        dq = _euler_to_quat(
            command[3] * self.cartesian_rotational_scale,
            command[4] * self.cartesian_rotational_scale,
            command[5] * self.cartesian_rotational_scale,
        )
        target.orientation = _quat_mul(pose.orientation, dq)
        return target

    def _start_cartesian_step(self, command):
        if not self.cartesian_path_client.service_is_ready():
            self.get_logger().warn(
                'compute_cartesian_path not ready — is move_group running?',
                throttle_duration_sec=3.0,
            )
            return

        current_pose = self._get_ee_pose()
        if current_pose is None:
            return

        target_pose = self._apply_delta(current_pose, command)

        req = GetCartesianPath.Request()
        req.header.frame_id = self.planning_frame
        req.header.stamp = self.get_clock().now().to_msg()
        req.group_name = self.arm_group_name
        req.link_name = self.ee_link
        req.waypoints = [target_pose]
        req.max_step = self.cartesian_max_step
        req.jump_threshold = 0.0
        req.avoid_collisions = False

        self.cartesian_executing = True
        self.cartesian_path_client.call_async(req).add_done_callback(self._on_path_computed)

    def _on_path_computed(self, future):
        try:
            resp = future.result()
        except Exception as e:
            self.get_logger().warn(f'Cartesian path error: {e}')
            self.cartesian_executing = False
            return

        if resp.fraction < self.cartesian_fraction_threshold:
            self.get_logger().info(
                f'Path fraction {resp.fraction:.2f} too low — near joint limit or unreachable pose',
                throttle_duration_sec=1.0,
            )
            self.cartesian_executing = False
            return

        if not self.execute_action.server_is_ready():
            self.get_logger().warn('execute_trajectory action not ready')
            self.cartesian_executing = False
            return

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = resp.solution
        self.execute_action.send_goal_async(goal).add_done_callback(self._on_goal_accepted)

    def _on_goal_accepted(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.cartesian_executing = False
            return
        goal_handle.get_result_async().add_done_callback(self._on_execute_done)

    def _on_execute_done(self, _future):
        self.cartesian_executing = False

    # ── Joint / Gripper actions ───────────────────────────────────────────────

    def send_arm_action(self, targets, duration=None, force=False):
        now = self.get_clock().now()
        if self.last_arm_action_send_time is not None and not force:
            if (now - self.last_arm_action_send_time) < Duration(seconds=self.trajectory_duration * 0.8):
                return False
        if not self.arm_action.server_is_ready():
            if not self.arm_action.wait_for_server(timeout_sec=0.1):
                return False
        dur = self.trajectory_duration if duration is None else duration
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.ARM_JOINTS
        pt = JointTrajectoryPoint()
        pt.positions = targets
        pt.velocities = [0.0] * len(self.ARM_JOINTS)
        pt.time_from_start.sec = int(dur)
        pt.time_from_start.nanosec = int((dur - int(dur)) * 1e9)
        goal.trajectory.points = [pt]
        self.last_arm_action_send_time = now
        self.arm_action.send_goal_async(goal)
        return True

    def send_gripper_action(self, target):
        now = self.get_clock().now()
        if self.last_gripper_goal_position is not None:
            if abs(target - self.last_gripper_goal_position) <= self.gripper_command_epsilon:
                return
        if self.last_gripper_action_send_time is not None:
            if (now - self.last_gripper_action_send_time) < Duration(seconds=self.trajectory_duration * 0.8):
                return
        if not self.gripper_action.server_is_ready():
            if not self.gripper_action.wait_for_server(timeout_sec=0.1):
                return
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [self.GRIPPER_JOINT]
        pt = JointTrajectoryPoint()
        pt.positions = [target]
        pt.velocities = [0.0]
        pt.time_from_start.sec = int(self.trajectory_duration)
        pt.time_from_start.nanosec = int((self.trajectory_duration - int(self.trajectory_duration)) * 1e9)
        goal.trajectory.points = [pt]
        self.last_gripper_action_send_time = now
        self.last_gripper_goal_position = target
        self.gripper_action.send_goal_async(goal)


def main(args=None):
    rclpy.init(args=args)
    node = PiperVelocityTeleop()
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
