import os
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro


def _prefix_piper_model(robot_xml, prefix):
    for link in robot_xml.findall('link'):
        link.set('name', prefix + link.get('name'))

    for joint in robot_xml.findall('joint'):
        joint.set('name', prefix + joint.get('name'))

        parent = joint.find('parent')
        if parent is not None and parent.get('link'):
            parent.set('link', prefix + parent.get('link'))

        child = joint.find('child')
        if child is not None and child.get('link'):
            child.set('link', prefix + child.get('link'))

    for transmission in robot_xml.findall('transmission'):
        if transmission.get('name'):
            transmission.set('name', prefix + transmission.get('name'))

        for joint in transmission.findall('joint'):
            if joint.get('name'):
                joint.set('name', prefix + joint.get('name'))

        for actuator in transmission.findall('actuator'):
            if actuator.get('name'):
                actuator.set('name', prefix + actuator.get('name'))

    for gazebo in robot_xml.findall('gazebo'):
        if gazebo.get('reference'):
            gazebo.set('reference', prefix + gazebo.get('reference'))

    for ros2_control in robot_xml.findall('ros2_control'):
        for joint in ros2_control.findall('joint'):
            if joint.get('name'):
                joint.set('name', prefix + joint.get('name'))


def _fix_nixito_mesh_paths(robot_xml):
    for mesh in robot_xml.iter('mesh'):
        filename = mesh.get('filename')
        if filename and filename.startswith('package://') and '/meshes/' in filename:
            mesh_file = filename.split('/meshes/', 1)[1]
            mesh.set('filename', f'package://nixito_description/meshes/{mesh_file}')


def _build_combined_robot_description(context):
    nixito_urdf_file = LaunchConfiguration('nixito_urdf_file').perform(context)
    piper_prefix = LaunchConfiguration('piper_prefix').perform(context)
    piper_parent_link = LaunchConfiguration('piper_parent_link').perform(context)
    piper_xyz = LaunchConfiguration('piper_xyz').perform(context)
    piper_rpy = LaunchConfiguration('piper_rpy').perform(context)
    nixito_y_offset = LaunchConfiguration('nixito_y_offset').perform(context)

    nixito_description_path = get_package_share_directory('nixito_description')
    piper_moveit_path = get_package_share_directory('piper_with_gripper_moveit')
    nixito_urdf = os.path.join(
        nixito_description_path,
        'urdf',
        nixito_urdf_file,
    )
    piper_xacro = os.path.join(piper_moveit_path, 'config', 'piper.urdf.xacro')
    piper_initial_positions = os.path.join(
        piper_moveit_path,
        'config',
        'initial_positions.yaml',
    )

    nixito_robot = ET.parse(nixito_urdf).getroot()
    piper_robot = ET.fromstring(
        xacro.process_file(
            piper_xacro,
            mappings={'initial_positions_file': piper_initial_positions},
        ).toxml()
    )

    _fix_nixito_mesh_paths(nixito_robot)
    _prefix_piper_model(piper_robot, piper_prefix)

    nixito_robot.set('name', 'nixito_2026_with_piper')

    mount_link = ET.Element('link', {'name': 'nixito_mount_link'})
    centering_joint = ET.Element('joint', {'name': 'nixito_centering_joint', 'type': 'fixed'})
    ET.SubElement(centering_joint, 'origin', {'xyz': f'0 {nixito_y_offset} 0', 'rpy': '0 0 0'})
    ET.SubElement(centering_joint, 'parent', {'link': 'nixito_mount_link'})
    ET.SubElement(centering_joint, 'child', {'link': 'base_link'})

    nixito_robot.insert(0, centering_joint)
    nixito_robot.insert(0, mount_link)

    mount_joint = ET.Element('joint', {'name': 'nixito_to_piper_mount_joint', 'type': 'fixed'})
    ET.SubElement(mount_joint, 'origin', {'xyz': piper_xyz, 'rpy': piper_rpy})
    ET.SubElement(mount_joint, 'parent', {'link': piper_parent_link})
    ET.SubElement(mount_joint, 'child', {'link': piper_prefix + 'world'})
    nixito_robot.append(mount_joint)

    for child in list(piper_robot):
        nixito_robot.append(child)

    return ET.tostring(nixito_robot, encoding='unicode')


def _launch_setup(context, *args, **kwargs):
    robot_description = _build_combined_robot_description(context)
    rviz_config = os.path.join(
        get_package_share_directory('nixito_description'),
        'rviz',
        'nixito_piper.rviz',
    )

    return [
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}],
        ),
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            output='screen',
            parameters=[{'robot_description': robot_description}],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'nixito_urdf_file',
            default_value='NIXITO 2026 Robocup.SLDASM.urdf',
            description='Archivo URDF de NIXITO dentro de nixito_description/urdf.',
        ),
        DeclareLaunchArgument(
            'piper_prefix',
            default_value='piper_',
            description='Prefijo aplicado a todos los links y joints del brazo Piper.',
        ),
        DeclareLaunchArgument(
            'piper_parent_link',
            default_value='nixito_mount_link',
            description='Link de NIXITO donde se monta la base del Piper.',
        ),
        DeclareLaunchArgument(
            'piper_xyz',
            default_value='0 -0.137 0.209',
            description='Posicion xyz del Piper respecto al link padre de NIXITO.',
        ),
        DeclareLaunchArgument(
            'piper_rpy',
            default_value='0 0 -1.5708',
            description='Orientacion rpy del Piper respecto al link padre de NIXITO.',
        ),
        DeclareLaunchArgument(
            'nixito_y_offset',
            default_value='0.0',
            description='Correccion en Y para centrar el URDF exportado de NIXITO respecto al Piper.',
        ),
        OpaqueFunction(function=_launch_setup),
    ])
