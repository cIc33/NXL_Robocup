#!/usr/bin/env python3
import math
import os

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose, Quaternion
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import BoundingVolume, Constraints, JointConstraint, OrientationConstraint, PositionConstraint
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String


class PiperNamedTargetExecutor(Node):
    def __init__(self):
        super().__init__('piper_named_target_executor')

        self.declare_parameter('config_file', 'named_targets.yaml')
        self.declare_parameter('command_topic', '/piper/named_target')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('move_action_name', '/move_action')
        self.declare_parameter('require_recent_joint_state', True)
        self.declare_parameter('joint_state_timeout', 2.0)
        self.declare_parameter('default_planning_time', 5.0)
        self.declare_parameter('default_planning_attempts', 1)
        self.declare_parameter('default_velocity_scaling', 0.2)
        self.declare_parameter('default_acceleration_scaling', 0.2)

        self.command_topic = str(self.get_parameter('command_topic').value)
        self.joint_states_topic = str(self.get_parameter('joint_states_topic').value)
        self.move_action_name = str(self.get_parameter('move_action_name').value)
        self.require_recent_joint_state = bool(self.get_parameter('require_recent_joint_state').value)
        self.joint_state_timeout = float(self.get_parameter('joint_state_timeout').value)
        self.default_planning_time = float(self.get_parameter('default_planning_time').value)
        self.default_planning_attempts = int(self.get_parameter('default_planning_attempts').value)
        self.default_velocity_scaling = float(self.get_parameter('default_velocity_scaling').value)
        self.default_acceleration_scaling = float(self.get_parameter('default_acceleration_scaling').value)

        config_file = str(self.get_parameter('config_file').value)
        self.targets = self.load_targets(config_file)

        self.busy = False
        self.last_joint_state_time = None
        self.action_client = ActionClient(self, MoveGroup, self.move_action_name)
        self.create_subscription(String, self.command_topic, self.target_callback, 10)
        self.create_subscription(JointState, self.joint_states_topic, self.joint_state_callback, 10)

        self.get_logger().info(
            f'Named target executor ready on {self.command_topic}. Targets: '
            + ', '.join(sorted(self.targets.keys()))
        )

    def load_targets(self, config_file):
        if os.path.isabs(config_file):
            config_path = config_file
        else:
            config_path = os.path.join(
                get_package_share_directory('piper_with_gripper_moveit'),
                'config',
                config_file,
            )

        with open(config_path, 'r', encoding='utf-8') as handle:
            data = yaml.safe_load(handle) or {}
        targets = data.get('targets', {})
        if not targets:
            raise ValueError(f'No targets found in {config_path}')
        return targets

    def joint_state_callback(self, msg: JointState):
        if msg.header.stamp.sec == 0 and msg.header.stamp.nanosec == 0:
            return
        self.last_joint_state_time = self.get_clock().now()

    def has_recent_joint_state(self):
        if not self.require_recent_joint_state:
            return True
        if self.last_joint_state_time is None:
            return False
        age = (self.get_clock().now() - self.last_joint_state_time).nanoseconds / 1e9
        return age < self.joint_state_timeout

    def target_callback(self, msg: String):
        target_name = msg.data.strip()
        if not target_name:
            return
        if target_name not in self.targets:
            self.get_logger().error(f"Unknown target '{target_name}'")
            return
        if self.busy:
            self.get_logger().warn(f"Ignoring target '{target_name}' because a motion is already active")
            return
        if not self.has_recent_joint_state():
            self.get_logger().warn(
                f"Ignoring target '{target_name}' because {self.joint_states_topic} is not ready or is stale"
            )
            return
        if not self.action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(f'MoveGroup action server is not available on {self.move_action_name}')
            return

        target = self.targets[target_name]
        try:
            goal = self.build_goal(target_name, target)
        except (KeyError, TypeError, ValueError) as exc:
            self.get_logger().error(f"Invalid target '{target_name}': {exc}")
            return

        self.busy = True
        self.get_logger().info(f"Sending target '{target_name}' ({self.target_type(target)})")
        send_goal_future = self.action_client.send_goal_async(goal)
        send_goal_future.add_done_callback(lambda future, name=target_name: self.goal_response_callback(future, name))

    def build_goal(self, target_name, target):
        goal = MoveGroup.Goal()
        goal.request.group_name = str(target['group'])
        goal.request.num_planning_attempts = int(target.get('planning_attempts', self.default_planning_attempts))
        goal.request.allowed_planning_time = float(target.get('planning_time', self.default_planning_time))
        goal.request.max_velocity_scaling_factor = float(
            target.get('velocity_scaling', self.default_velocity_scaling)
        )
        goal.request.max_acceleration_scaling_factor = float(
            target.get('acceleration_scaling', self.default_acceleration_scaling)
        )
        goal.request.start_state.is_diff = True

        constraint = Constraints()
        constraint.name = target_name
        target_type = self.target_type(target)
        if target_type == 'joint':
            constraint.joint_constraints = self.build_joint_constraints(target)
        elif target_type == 'cartesian':
            position_constraint, orientation_constraint = self.build_cartesian_constraints(target)
            constraint.position_constraints.append(position_constraint)
            if orientation_constraint is not None:
                constraint.orientation_constraints.append(orientation_constraint)
        else:
            raise ValueError("type must be 'joint' or 'cartesian'")

        goal.request.goal_constraints.append(constraint)
        goal.planning_options.plan_only = bool(target.get('plan_only', False))
        goal.planning_options.look_around = False
        goal.planning_options.replan = bool(target.get('replan', False))
        return goal

    def target_type(self, target):
        if 'type' in target:
            return str(target['type']).lower()
        if 'joints' in target:
            return 'joint'
        if 'position' in target or 'pose' in target:
            return 'cartesian'
        return 'unknown'

    def build_joint_constraints(self, target):
        joints = target['joints']
        tolerance = float(target.get('tolerance', 0.001))
        constraints = []
        for joint_name, position in joints.items():
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = str(joint_name)
            joint_constraint.position = float(position)
            joint_constraint.tolerance_above = tolerance
            joint_constraint.tolerance_below = tolerance
            joint_constraint.weight = 1.0
            constraints.append(joint_constraint)
        return constraints

    def build_cartesian_constraints(self, target):
        pose_data = target.get('pose', target)
        frame_id = str(pose_data.get('frame_id', target.get('frame_id', 'base_link')))
        link_name = str(pose_data.get('link_name', target.get('link_name', 'gripper_base')))
        position = self.read_xyz(pose_data['position'])

        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = frame_id
        position_constraint.link_name = link_name
        position_constraint.weight = 1.0

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [float(pose_data.get('position_tolerance', target.get('position_tolerance', 0.01)))]

        region_pose = Pose()
        region_pose.position.x = position[0]
        region_pose.position.y = position[1]
        region_pose.position.z = position[2]
        region_pose.orientation.w = 1.0

        region = BoundingVolume()
        region.primitives.append(primitive)
        region.primitive_poses.append(region_pose)
        position_constraint.constraint_region = region

        quaternion = self.read_orientation(pose_data)
        if quaternion is None:
            return position_constraint, None

        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = frame_id
        orientation_constraint.link_name = link_name
        orientation_constraint.orientation = quaternion
        tolerance = float(pose_data.get('orientation_tolerance', target.get('orientation_tolerance', 0.05)))
        orientation_constraint.absolute_x_axis_tolerance = tolerance
        orientation_constraint.absolute_y_axis_tolerance = tolerance
        orientation_constraint.absolute_z_axis_tolerance = tolerance
        orientation_constraint.weight = 1.0
        return position_constraint, orientation_constraint

    def read_xyz(self, data):
        if isinstance(data, dict):
            return [float(data['x']), float(data['y']), float(data['z'])]
        if isinstance(data, (list, tuple)) and len(data) == 3:
            return [float(data[0]), float(data[1]), float(data[2])]
        raise ValueError('position must be {x, y, z} or [x, y, z]')

    def read_orientation(self, data):
        if 'quaternion' in data:
            quaternion = data['quaternion']
            if isinstance(quaternion, dict):
                return Quaternion(
                    x=float(quaternion['x']),
                    y=float(quaternion['y']),
                    z=float(quaternion['z']),
                    w=float(quaternion['w']),
                )
            if isinstance(quaternion, (list, tuple)) and len(quaternion) == 4:
                return Quaternion(
                    x=float(quaternion[0]),
                    y=float(quaternion[1]),
                    z=float(quaternion[2]),
                    w=float(quaternion[3]),
                )
            raise ValueError('quaternion must be {x, y, z, w} or [x, y, z, w]')

        rpy = data.get('rpy', data.get('orientation_rpy'))
        if rpy is None:
            return None
        if isinstance(rpy, dict):
            roll = float(rpy['roll'])
            pitch = float(rpy['pitch'])
            yaw = float(rpy['yaw'])
        elif isinstance(rpy, (list, tuple)) and len(rpy) == 3:
            roll = float(rpy[0])
            pitch = float(rpy[1])
            yaw = float(rpy[2])
        else:
            raise ValueError('rpy must be {roll, pitch, yaw} or [roll, pitch, yaw]')
        return self.quaternion_from_rpy(roll, pitch, yaw)

    def quaternion_from_rpy(self, roll, pitch, yaw):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        return Quaternion(
            x=sr * cp * cy - cr * sp * sy,
            y=cr * sp * cy + sr * cp * sy,
            z=cr * cp * sy - sr * sp * cy,
            w=cr * cp * cy + sr * sp * sy,
        )

    def goal_response_callback(self, future, target_name):
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.busy = False
            self.get_logger().error(f"Target '{target_name}' was rejected by MoveGroup")
            return
        self.get_logger().info(f"Target '{target_name}' accepted")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda future, name=target_name: self.result_callback(future, name))

    def result_callback(self, future, target_name):
        self.busy = False
        result = future.result().result
        error_code = result.error_code.val
        if error_code == 1:
            self.get_logger().info(f"Target '{target_name}' completed successfully")
        else:
            self.get_logger().error(f"Target '{target_name}' failed with MoveIt error code {error_code}")


def main(args=None):
    rclpy.init(args=args)
    node = PiperNamedTargetExecutor()
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
