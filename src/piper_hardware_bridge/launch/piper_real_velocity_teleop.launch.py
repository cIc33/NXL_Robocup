from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    bridge_pkg = get_package_share_directory('piper_hardware_bridge')

    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bridge_pkg, 'launch', 'piper_real_bringup.launch.py')
        )
    )

    velocity_bridge = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='piper_hardware_bridge',
                executable='piper_hardware_velocity_bridge',
                name='piper_hardware_velocity_bridge',
                output='screen',
                parameters=[{
                    'command_topic': '/piper/test_velocity_cmd',
                    'switch_mode_topic': '/piper/switch_mode',
                    'joint_states_topic': '/joint_states',
                    'joint_command_topic': '/piper/joint_ctrl_cmd',
                    'enable_topic': '/piper/enable',
                    'auto_enable_on_start': False,
                }],
            )
        ],
    )

    return LaunchDescription([
        bringup,
        velocity_bridge,
    ])
