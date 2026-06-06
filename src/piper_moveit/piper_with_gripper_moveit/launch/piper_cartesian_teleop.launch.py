import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('piper_with_gripper_moveit')

    demo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg, 'launch', 'demo.launch.py')
        )
    )

    teleop = TimerAction(
        period=7.0,
        actions=[
            Node(
                package='piper_with_gripper_moveit',
                executable='piper_velocity_teleop.py',
                output='screen',
            )
        ],
    )

    named_targets = TimerAction(
        period=7.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(pkg, 'launch', 'piper_named_targets.launch.py')
                )
            )
        ],
    )

    return LaunchDescription([
        demo,
        teleop,
        named_targets,
    ])
