from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('config_file', default_value='named_targets.yaml'),
        DeclareLaunchArgument('command_topic', default_value='/piper/named_target'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        Node(
            package='piper_with_gripper_moveit',
            executable='piper_named_target_executor.py',
            output='screen',
            parameters=[{
                'config_file': LaunchConfiguration('config_file'),
                'command_topic': LaunchConfiguration('command_topic'),
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            }],
        ),
    ])
