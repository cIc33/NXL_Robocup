from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    can_port_arg = DeclareLaunchArgument('can_port', default_value='can0')
    auto_enable_arg = DeclareLaunchArgument('auto_enable', default_value='false')
    gripper_exist_arg = DeclareLaunchArgument('gripper_exist', default_value='true')
    gripper_val_mutiple_arg = DeclareLaunchArgument('gripper_val_mutiple', default_value='1')
    log_level_arg = DeclareLaunchArgument('log_level', default_value='info')

    piper_driver = Node(
        package='piper',
        executable='piper_single_ctrl',
        name='piper_ctrl_single_node',
        output='screen',
        ros_arguments=['--log-level', LaunchConfiguration('log_level')],
        parameters=[{
            'can_port': LaunchConfiguration('can_port'),
            'auto_enable': LaunchConfiguration('auto_enable'),
            'gripper_exist': LaunchConfiguration('gripper_exist'),
            'gripper_val_mutiple': LaunchConfiguration('gripper_val_mutiple'),
        }],
        remappings=[
            ('joint_ctrl_single', '/piper/joint_ctrl_cmd'),
            ('pos_cmd', '/piper/pos_cmd'),
            ('enable_flag', '/piper/enable'),
            ('joint_states_single', '/piper/joint_states_single'),
            ('joint_states_feedback', '/piper/joint_states_feedback'),
            ('joint_ctrl', '/piper/joint_ctrl_feedback'),
            ('arm_status', '/piper/arm_status'),
            ('end_pose', '/piper/end_pose'),
            ('end_pose_stamped', '/piper/end_pose_stamped'),
        ],
    )

    normalizer = Node(
        package='piper_hardware_bridge',
        executable='piper_joint_state_normalizer',
        name='piper_joint_state_normalizer',
        output='screen',
        parameters=[{
            'input_topic': '/piper/joint_states_feedback',
            'output_topic': '/joint_states',
            'include_gripper': ParameterValue(LaunchConfiguration('gripper_exist'), value_type=bool),
        }],
    )

    return LaunchDescription([
        can_port_arg,
        auto_enable_arg,
        gripper_exist_arg,
        gripper_val_mutiple_arg,
        log_level_arg,
        piper_driver,
        normalizer,
    ])
