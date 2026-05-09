import os
import re

from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

import xacro


def remove_comments(text):
    pattern = r'<!--(.*?)-->'
    return re.sub(pattern, '', text, flags=re.DOTALL)


def generate_launch_description():
    robot_name_in_model = 'piper_x'
    package_name = 'piper_x_gazebo'
    urdf_name = 'piper_x_gazebo.urdf.xacro'
    world_name = 'piper_x_with_estop.world'

    pkg_share = FindPackageShare(package=package_name).find(package_name)
    perception_share = FindPackageShare(package='piper_perception').find('piper_perception')
    urdf_model_path = os.path.join(pkg_share, f'config/{urdf_name}')
    world_path = os.path.join(pkg_share, f'worlds/{world_name}')
    detector_launch_path = os.path.join(perception_share, 'launch', 'detector.launch.py')

    start_gazebo_cmd = ExecuteProcess(
        cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so',
             world_path],
        output='screen',
    )

    doc = xacro.parse(open(urdf_model_path))
    xacro.process_doc(doc)
    params = {'robot_description': remove_comments(doc.toxml())}

    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'use_sim_time': True}, params, {'publish_frequency': 15.0}],
        output='screen',
    )

    spawn_entity_cmd = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', robot_name_in_model, '-topic', 'robot_description'],
        output='screen',
    )

    load_joint_state_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'joint_state_broadcaster'],
        output='screen',
    )

    load_joint_trajectory_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'arm_controller'],
        output='screen',
    )

    load_gripper_trajectory_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'gripper_controller'],
        output='screen',
    )

    close_evt1 = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_entity_cmd,
            on_exit=[load_joint_state_controller],
        )
    )

    close_evt2 = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=load_joint_state_controller,
            on_exit=[load_joint_trajectory_controller, load_gripper_trajectory_controller],
        )
    )

    detector_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(detector_launch_path),
        launch_arguments={'use_sim_time': 'true'}.items(),
    )

    ld = LaunchDescription()
    ld.add_action(close_evt1)
    ld.add_action(close_evt2)
    ld.add_action(start_gazebo_cmd)
    ld.add_action(node_robot_state_publisher)
    ld.add_action(spawn_entity_cmd)
    ld.add_action(detector_launch)
    return ld
