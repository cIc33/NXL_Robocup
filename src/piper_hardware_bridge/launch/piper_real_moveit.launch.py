from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    bridge_pkg = get_package_share_directory('piper_hardware_bridge')
    moveit_pkg = get_package_share_directory('piper_with_gripper_moveit')

    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bridge_pkg, 'launch', 'piper_real_bringup.launch.py')
        )
    )

    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(moveit_pkg, 'launch', 'rsp.launch.py')
        )
    )

    trajectory_bridge = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='piper_hardware_bridge',
                executable='piper_follow_joint_trajectory_bridge',
                name='piper_follow_joint_trajectory_bridge',
                output='screen',
                parameters=[{
                    'joint_states_topic': '/joint_states',
                    'joint_command_topic': '/piper/joint_ctrl_cmd',
                    'arm_action_name': '/arm_controller/follow_joint_trajectory',
                    'gripper_action_name': '/gripper_controller/follow_joint_trajectory',
                }],
            )
        ],
    )

    move_group = TimerAction(
        period=3.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(moveit_pkg, 'launch', 'move_group.launch.py')
                )
            )
        ],
    )

    rviz = TimerAction(
        period=4.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(moveit_pkg, 'launch', 'moveit_rviz.launch.py')
                )
            )
        ],
    )

    return LaunchDescription([
        bringup,
        robot_state_publisher,
        trajectory_bridge,
        move_group,
        rviz,
    ])
