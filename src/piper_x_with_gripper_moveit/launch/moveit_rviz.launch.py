from moveit_configs_utils import MoveItConfigsBuilder

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder(
        'piper_x',
        package_name='piper_x_with_gripper_moveit',
    ).to_moveit_configs()

    ld = LaunchDescription()
    ld.add_action(DeclareLaunchArgument('rviz_config', default_value=str(moveit_config.package_path / 'config/moveit.rviz')))

    rviz_parameters = [
        moveit_config.robot_description,
        moveit_config.robot_description_semantic,
        moveit_config.planning_pipelines,
        moveit_config.robot_description_kinematics,
        moveit_config.joint_limits,
    ]

    ld.add_action(
        Node(
            package='rviz2',
            executable='rviz2',
            output='log',
            respawn=False,
            arguments=['-d', LaunchConfiguration('rviz_config')],
            parameters=rviz_parameters,
        )
    )

    return ld
