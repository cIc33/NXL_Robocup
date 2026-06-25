import os
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, TimerAction, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder
import xacro


def _fix_nixito_mesh_paths(robot_xml):
    for mesh in robot_xml.iter('mesh'):
        filename = mesh.get('filename')
        if filename and filename.startswith('package://') and '/meshes/' in filename:
            mesh_file = filename.split('/meshes/', 1)[1]
            mesh.set('filename', f'package://nixito_description/meshes/{mesh_file}')


def _remove_children(element, child_tags):
    for child in list(element):
        if child.tag in child_tags:
            element.remove(child)


def _prefix_nixito_model(robot_xml, prefix):
    movable_joints = {'flipper_R_joint', 'flipper_L_joint'}

    for link in robot_xml.findall('link'):
        link.set('name', prefix + link.get('name'))

    for joint in robot_xml.findall('joint'):
        original_name = joint.get('name')
        joint.set('name', prefix + original_name)

        parent = joint.find('parent')
        if parent is not None and parent.get('link'):
            parent.set('link', prefix + parent.get('link'))

        child = joint.find('child')
        if child is not None and child.get('link'):
            child.set('link', prefix + child.get('link'))

        if original_name not in movable_joints:
            joint.set('type', 'fixed')
            _remove_children(joint, {'axis', 'limit', 'dynamics', 'mimic', 'safety_controller', 'calibration'})


def _build_combined_robot_description(context):
    nixito_urdf_file = LaunchConfiguration('nixito_urdf_file').perform(context)
    nixito_prefix = LaunchConfiguration('nixito_prefix').perform(context)
    nixito_y_offset = LaunchConfiguration('nixito_y_offset').perform(context)
    piper_xyz = LaunchConfiguration('piper_xyz').perform(context)
    piper_rpy = LaunchConfiguration('piper_rpy').perform(context)

    nixito_description_path = get_package_share_directory('nixito_description')
    piper_moveit_path = get_package_share_directory('piper_with_gripper_moveit')

    nixito_urdf = os.path.join(nixito_description_path, 'urdf', nixito_urdf_file)
    piper_xacro = os.path.join(piper_moveit_path, 'config', 'piper.urdf.xacro')
    piper_initial_positions = os.path.join(piper_moveit_path, 'config', 'initial_positions.yaml')

    nixito_robot = ET.parse(nixito_urdf).getroot()
    piper_robot = ET.fromstring(
        xacro.process_file(
            piper_xacro,
            mappings={'initial_positions_file': piper_initial_positions},
        ).toxml()
    )

    _fix_nixito_mesh_paths(nixito_robot)
    _prefix_nixito_model(nixito_robot, nixito_prefix)

    nixito_robot.set('name', 'piper')

    mount_link_name = nixito_prefix + 'mount_link'
    nixito_base_link_name = nixito_prefix + 'base_link'
    centering_joint_name = nixito_prefix + 'centering_joint'
    piper_mount_joint_name = nixito_prefix + 'to_piper_mount_joint'

    mount_link = ET.Element('link', {'name': mount_link_name})
    centering_joint = ET.Element('joint', {'name': centering_joint_name, 'type': 'fixed'})
    ET.SubElement(centering_joint, 'origin', {'xyz': f'0 {nixito_y_offset} 0', 'rpy': '0 0 0'})
    ET.SubElement(centering_joint, 'parent', {'link': mount_link_name})
    ET.SubElement(centering_joint, 'child', {'link': nixito_base_link_name})

    piper_mount_joint = ET.Element('joint', {'name': piper_mount_joint_name, 'type': 'fixed'})
    ET.SubElement(piper_mount_joint, 'origin', {'xyz': piper_xyz, 'rpy': piper_rpy})
    ET.SubElement(piper_mount_joint, 'parent', {'link': mount_link_name})
    ET.SubElement(piper_mount_joint, 'child', {'link': 'world'})

    nixito_robot.insert(0, centering_joint)
    nixito_robot.insert(0, mount_link)
    nixito_robot.append(piper_mount_joint)

    for child in list(piper_robot):
        nixito_robot.append(child)

    return ET.tostring(nixito_robot, encoding='unicode')


def _launch_setup(context, *args, **kwargs):
    robot_description = _build_combined_robot_description(context)
    bridge_pkg = get_package_share_directory('piper_hardware_bridge')
    moveit_pkg = get_package_share_directory('piper_with_gripper_moveit')
    rviz_config = os.path.join(
        get_package_share_directory('nixito_description'),
        'rviz',
        'nixito_piper_moveit.rviz',
    )

    moveit_config = MoveItConfigsBuilder('piper', package_name='piper_with_gripper_moveit').to_moveit_configs()

    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(bridge_pkg, 'launch', 'piper_real_bringup.launch.py')),
        launch_arguments={
            'can_port': LaunchConfiguration('can_port'),
            'auto_enable': LaunchConfiguration('auto_enable'),
            'gripper_exist': LaunchConfiguration('gripper_exist'),
            'gripper_val_mutiple': LaunchConfiguration('gripper_val_mutiple'),
            'log_level': LaunchConfiguration('log_level'),
        }.items(),
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}],
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

    move_group = TimerAction(
        period=3.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(moveit_pkg, 'launch', 'move_group.launch.py')),
            )
        ],
    )

    enable_arm = TimerAction(
        period=7.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'topic', 'pub', '--once',
                    '/piper/enable', 'std_msgs/msg/Bool',
                    'data: true',
                ],
                output='screen',
            )
        ],
    )

    rviz = TimerAction(
        period=4.0,
        actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                output='screen',
                arguments=['-d', rviz_config],
                parameters=[
                    moveit_config.robot_description,
                    moveit_config.robot_description_semantic,
                    moveit_config.robot_description_kinematics,
                    moveit_config.planning_pipelines,
                ],
            )
        ],
    )

    named_targets = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='piper_with_gripper_moveit',
                executable='piper_named_target_executor.py',
                name='piper_named_target_executor',
                output='screen',
                parameters=[{
                    'config_file': os.path.join(moveit_pkg, 'config', 'named_targets.yaml'),
                    'command_topic': '/piper/named_target',
                }],
            )
        ],
    )

    return [bringup, robot_state_publisher, trajectory_bridge, velocity_bridge, enable_arm, move_group, rviz, named_targets]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('can_port', default_value='can0'),
        DeclareLaunchArgument('auto_enable', default_value='false'),
        DeclareLaunchArgument('gripper_exist', default_value='true'),
        DeclareLaunchArgument('gripper_val_mutiple', default_value='1'),
        DeclareLaunchArgument('log_level', default_value='info'),
        DeclareLaunchArgument(
            'nixito_urdf_file',
            default_value='NIXITO 2026 Robocup.SLDASM.urdf',
            description='Archivo URDF de NIXITO dentro de nixito_description/urdf.',
        ),
        DeclareLaunchArgument(
            'nixito_prefix',
            default_value='nixito/',
            description='Prefijo aplicado a todos los links y joints de NIXITO.',
        ),
        DeclareLaunchArgument(
            'piper_xyz',
            default_value='0 -0.137 0.209',
            description='Posicion xyz del Piper respecto a nixito/mount_link.',
        ),
        DeclareLaunchArgument(
            'piper_rpy',
            default_value='0 0 -1.5708',
            description='Orientacion rpy del Piper respecto a nixito/mount_link.',
        ),
        DeclareLaunchArgument(
            'nixito_y_offset',
            default_value='0.0',
            description='Correccion en Y para centrar el URDF exportado de NIXITO.',
        ),
        OpaqueFunction(function=_launch_setup),
    ])
