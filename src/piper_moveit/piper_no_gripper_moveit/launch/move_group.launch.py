import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch
from launch_ros.actions import Node

def generate_launch_description():
    pkg = get_package_share_directory("piper_no_gripper_moveit")
    
    servo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, "launch", "piper_servo.launch.py"))
    )
    
    teleop = Node(
        package="piper_with_gripper_moveit",
        executable="piper_velocity_teleop",
        output="screen",
    )

    return LaunchDescription([
        servo,
        teleop,
    ])
    