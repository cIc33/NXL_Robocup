from moveit_configs_utils import MoveItConfigsBuilder

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from moveit_configs_utils.launch_utils import DeclareBooleanLaunchArg, add_debuggable_node
from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder(
        'piper_x',
        package_name='piper_x_with_gripper_moveit',
    ).to_moveit_configs()

    ld = LaunchDescription()
    my_generate_move_group_launch(ld, moveit_config)
    my_generate_moveit_rviz_launch(ld, moveit_config)
    my_generate_press_estop_launch(ld)
    return ld


def my_generate_press_estop_launch(ld):
    press_estop_launch_path = os.path.join(
        get_package_share_directory('piper_l_tasking'),
        'launch',
        'press_estop.launch.py',
    )

    ld.add_action(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(press_estop_launch_path),
            launch_arguments={'use_sim_time': 'true'}.items(),
        )
    )
    return ld


def my_generate_move_group_launch(ld, moveit_config):
    ld.add_action(DeclareBooleanLaunchArg('debug', default_value=False))
    ld.add_action(DeclareBooleanLaunchArg('allow_trajectory_execution', default_value=True))
    ld.add_action(DeclareBooleanLaunchArg('publish_monitored_planning_scene', default_value=True))
    ld.add_action(DeclareLaunchArgument('capabilities', default_value=''))
    ld.add_action(DeclareLaunchArgument('disable_capabilities', default_value=''))
    ld.add_action(DeclareBooleanLaunchArg('monitor_dynamics', default_value=False))

    should_publish = LaunchConfiguration('publish_monitored_planning_scene')
    move_group_configuration = {
        'publish_robot_description': True,
        'publish_robot_description_semantic': True,
        'allow_trajectory_execution': LaunchConfiguration('allow_trajectory_execution'),
        'capabilities': ParameterValue(LaunchConfiguration('capabilities'), value_type=str),
        'disable_capabilities': ParameterValue(LaunchConfiguration('disable_capabilities'), value_type=str),
        'publish_planning_scene': should_publish,
        'publish_geometry_updates': should_publish,
        'publish_state_updates': should_publish,
        'publish_transforms_updates': should_publish,
        'monitor_dynamics': False,
    }

    move_group_params = [
        moveit_config.to_dict(),
        move_group_configuration,
        {'use_sim_time': True},
    ]

    add_debuggable_node(
        ld,
        package='moveit_ros_move_group',
        executable='move_group',
        commands_file=str(moveit_config.package_path / 'launch' / 'gdb_settings.gdb'),
        output='screen',
        parameters=move_group_params,
        extra_debug_args=['--debug'],
        additional_env={'DISPLAY': ':0'},
    )
    return ld


def my_generate_moveit_rviz_launch(ld, moveit_config):
    ld.add_action(DeclareBooleanLaunchArg('debug', default_value=False))
    ld.add_action(
        DeclareLaunchArgument(
            'rviz_config',
            default_value=str(moveit_config.package_path / 'config/moveit.rviz'),
        )
    )

    rviz_parameters = [
        moveit_config.robot_description,
        moveit_config.robot_description_semantic,
        moveit_config.planning_pipelines,
        moveit_config.robot_description_kinematics,
        moveit_config.joint_limits,
        {'use_sim_time': True},
    ]

    add_debuggable_node(
        ld,
        package='rviz2',
        executable='rviz2',
        output='log',
        respawn=False,
        arguments=['-d', LaunchConfiguration('rviz_config')],
        parameters=rviz_parameters,
    )

    return ld
