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
        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('driver_speed_percent', 20.0)
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
        self.create_subscription(JointState, self.joint_states_topic, self.joint_state_callback, 10)
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

        joint_name = goal.trajectory.joint_names[0] if goal.trajectory.joint_names else 'joint7'
        for point in goal.trajectory.points:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return result
            if not point.positions:
                continue
            gripper_position = max(0.0, point.positions[0])
            arm_positions = [self.current_state.get(name, 0.0) for name in self.ARM_JOINTS]
            self.publish_joint_command(arm_positions, gripper_position)
            sleep_time = self.duration_to_seconds(point.time_from_start)
            time.sleep(max(0.02, min(0.5, sleep_time)))

        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        result.error_string = f'Executed gripper trajectory for {joint_name}'
        goal_handle.succeed()
        return result

    def execute_points(self, goal_handle, joint_names, points, start_positions):
        previous_positions = list(start_positions)
        previous_time = 0.0

        for point in points:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return False
            if len(point.positions) < len(joint_names):
                goal_handle.abort()
                return False

            desired_by_name = {
                name: point.positions[index]
                for index, name in enumerate(joint_names)
                if index < len(point.positions)
            }
            target_positions = [desired_by_name.get(name, previous_positions[index]) for index, name in enumerate(self.ARM_JOINTS)]
            target_time = self.duration_to_seconds(point.time_from_start)
            segment_duration = max(0.0, target_time - previous_time)
            self.publish_interpolated_segment(previous_positions, target_positions, segment_duration)
            previous_positions = target_positions
            previous_time = target_time

        return True

    def publish_interpolated_segment(self, start_positions, target_positions, duration):
        if duration <= 0.0:
            self.publish_joint_command(target_positions, self.get_current_gripper_position())
            return

        steps = max(1, int(duration * self.publish_rate))
        sleep_time = duration / steps
        for step in range(1, steps + 1):
            ratio = step / steps
            positions = [
                start + (target - start) * ratio
                for start, target in zip(start_positions, target_positions)
            ]
            self.publish_joint_command(positions, self.get_current_gripper_position())
            time.sleep(sleep_time)

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
