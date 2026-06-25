#!/usr/bin/env python3
import time

from control_msgs.action import FollowJointTrajectory
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState


class PiperFollowJointTrajectoryBridge(Node):
    ARM_JOINTS = [f'joint{i}' for i in range(1, 7)]
    DRIVER_JOINT_NAMES = ARM_JOINTS + ['gripper']

    def __init__(self):
        super().__init__('piper_follow_joint_trajectory_bridge')

        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('joint_command_topic', '/piper/joint_ctrl_cmd')
        self.declare_parameter('arm_action_name', '/arm_controller/follow_joint_trajectory')
        self.declare_parameter('gripper_action_name', '/gripper_controller/follow_joint_trajectory')
        self.declare_parameter('publish_rate', 30.0)
        self.declare_parameter('driver_speed_percent', 60.0)
        self.declare_parameter('gripper_effort', 1.0)
        self.declare_parameter('goal_state_timeout', 2.0)

        self.joint_states_topic = str(self.get_parameter('joint_states_topic').value)
        self.joint_command_topic = str(self.get_parameter('joint_command_topic').value)
        self.arm_action_name = str(self.get_parameter('arm_action_name').value)
        self.gripper_action_name = str(self.get_parameter('gripper_action_name').value)
        self.publish_rate = float(self.get_parameter('publish_rate').value)
        self.driver_speed_percent = float(self.get_parameter('driver_speed_percent').value)
        self.gripper_effort = float(self.get_parameter('gripper_effort').value)
        self.goal_state_timeout = float(self.get_parameter('goal_state_timeout').value)

        self.current_state = {}
        self.held_gripper = 0.0
        self.create_subscription(JointState, self.joint_states_topic, self.joint_state_callback, 10)
        self.create_subscription(JointState, self.joint_command_topic, self.command_echo_callback, 10)
        self.command_publisher = self.create_publisher(JointState, self.joint_command_topic, 10)

        self.arm_server = ActionServer(
            self,
            FollowJointTrajectory,
            self.arm_action_name,
            execute_callback=self.execute_arm_goal,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )
        self.gripper_server = ActionServer(
            self,
            FollowJointTrajectory,
            self.gripper_action_name,
            execute_callback=self.execute_gripper_goal,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

        self.get_logger().info(
            f'FollowJointTrajectory bridge ready. Actions: {self.arm_action_name}, {self.gripper_action_name}; '
            f'output: {self.joint_command_topic}'
        )

    def joint_state_callback(self, msg: JointState):
        self.current_state = {
            name: msg.position[index]
            for index, name in enumerate(msg.name)
            if index < len(msg.position)
        }

    def command_echo_callback(self, msg: JointState):
        # Track the last gripper value actually sent to the driver (by any bridge).
        # Motor feedback is unreliable when the gripper is soft — using the commanded
        # value avoids closing the gripper to 0 when a trajectory starts.
        if len(msg.position) >= 7:
            self.held_gripper = float(msg.position[6])

    def goal_callback(self, _goal_request):
        return GoalResponse.ACCEPT

    def cancel_callback(self, _goal_handle):
        return CancelResponse.ACCEPT

    def execute_arm_goal(self, goal_handle):
        goal = goal_handle.request
        result = FollowJointTrajectory.Result()

        if not self.wait_for_state():
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = 'No /joint_states feedback available'
            goal_handle.abort()
            return result

        if not goal.trajectory.points:
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = 'Trajectory has no points'
            goal_handle.abort()
            return result

        if not set(self.ARM_JOINTS).issubset(set(goal.trajectory.joint_names)):
            result.error_code = FollowJointTrajectory.Result.INVALID_JOINTS
            result.error_string = f'Expected arm joints {self.ARM_JOINTS}'
            goal_handle.abort()
            return result

        start_positions = [self.current_state.get(name, 0.0) for name in self.ARM_JOINTS]
        ok = self.execute_points(goal_handle, goal.trajectory.joint_names, goal.trajectory.points, start_positions)
        if not ok:
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = 'Goal canceled or interrupted'
            return result

        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        goal_handle.succeed()
        return result

    def execute_gripper_goal(self, goal_handle):
        # Gripper is controlled exclusively via the velocity bridge GUI.
        # Accept MoveIt gripper goals immediately without executing them so MoveIt
        # doesn't interfere with the current gripper position.
        result = FollowJointTrajectory.Result()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        goal_handle.succeed()
        return result

    def execute_points(self, goal_handle, joint_names, points, start_positions):
        if not points:
            return True

        period = 1.0 / self.publish_rate
        total_duration = self.duration_to_seconds(points[-1].time_from_start)

        # Use the last commanded gripper value, not motor feedback. The gripper motor
        # goes soft when idle and its feedback drops to 0, which would cause the trajectory
        # to command GripperCtrl(0) and close the gripper.
        fixed_gripper = self.held_gripper

        joint_indices = {name: i for i, name in enumerate(joint_names)}
        waypoints = [(0.0, list(start_positions))]
        for point in points:
            if len(point.positions) < len(joint_names):
                goal_handle.abort()
                return False
            t = self.duration_to_seconds(point.time_from_start)
            positions = [
                point.positions[joint_indices[name]]
                if name in joint_indices and joint_indices[name] < len(point.positions)
                else start_positions[i]
                for i, name in enumerate(self.ARM_JOINTS)
            ]
            waypoints.append((t, positions))

        start_wall = time.monotonic()

        while True:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return False

            elapsed = time.monotonic() - start_wall
            positions = self._interpolate_at_time(waypoints, min(elapsed, total_duration))
            self.publish_joint_command(positions, fixed_gripper)

            if elapsed >= total_duration:
                break

            time.sleep(period)

        return True

    def _interpolate_at_time(self, waypoints, t):
        for i in range(1, len(waypoints)):
            t0, pos0 = waypoints[i - 1]
            t1, pos1 = waypoints[i]
            if t <= t1:
                if t1 <= t0:
                    return list(pos1)
                ratio = (t - t0) / (t1 - t0)
                return [p0 + (p1 - p0) * ratio for p0, p1 in zip(pos0, pos1)]
        return list(waypoints[-1][1])

    def publish_joint_command(self, arm_positions, gripper_position):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.DRIVER_JOINT_NAMES
        msg.position = list(arm_positions) + [float(gripper_position)]
        msg.velocity = [0.0] * 6 + [max(1.0, min(100.0, self.driver_speed_percent))]
        msg.effort = [0.0] * 6 + [max(0.5, min(3.0, self.gripper_effort))]
        self.command_publisher.publish(msg)

    def get_current_gripper_position(self):
        if 'gripper' in self.current_state:
            return self.current_state['gripper']
        if 'joint7' in self.current_state:
            return max(0.0, self.current_state['joint7'])
        return 0.0

    def wait_for_state(self):
        start = time.monotonic()
        while rclpy.ok() and not self.current_state:
            if time.monotonic() - start > self.goal_state_timeout:
                return False
            time.sleep(0.02)
        return True

    @staticmethod
    def duration_to_seconds(duration_msg):
        return float(duration_msg.sec) + float(duration_msg.nanosec) * 1e-9


def main(args=None):
    rclpy.init(args=args)
    node = PiperFollowJointTrajectoryBridge()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
