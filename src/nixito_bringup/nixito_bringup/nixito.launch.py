import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import xacro

def generate_launch_description():

    # 1. Configuración de rutas
    pkg_name = 'nixito_description' 
    urdf_file_name = 'nixito.SLDASM.urdf' 
    urdf_path = os.path.join(
        get_package_share_directory(pkg_name),
        'urdf',
        urdf_file_name)
    
    # 1. Configuración de rutas y variables
    pkg_name_brazo = 'nixito_description' 
    urdf_file_name_brazo = 'orion_arm.SLDASM.urdf'  
    ns = 'orion_arm' # Namespace unificado
    
    pkg_share = get_package_share_directory(pkg_name_brazo)
    params_file = os.path.join(pkg_share, 'config', 'arm_presets.yaml')
    
    urdf_path = os.path.join(
        get_package_share_directory(pkg_name),
        'urdf',
        urdf_file_name_brazo)

    # Procesar Xacro si es necesario (si es URDF puro, solo lee el archivo)
    robot_description_config = xacro.process_file(urdf_path)
    robot_desc = robot_description_config.toxml()

    return LaunchDescription([
        
        Node(
        package="usb_cam",
        executable="usb_cam_node_exe",
        name="usb_cam",
        output="screen",
        parameters=[{
            "video_device": "/dev/video0", # Cambiado de /dev/video4 a /dev/video0
            "image_width": 640,
            "image_height": 480,
            "pixel_format": "yuyv", 
            "frame_rate": 30.0,
            # Evita que intente publicar profundidad y sature la red/logs
            "image_transport_blacklist": ["compressedDepth", "theora"]
        }],
        ),
        

        # Publicador de estado del robot brazo orion
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            namespace=ns,
            parameters=[{
                'robot_description': robot_desc,
                'frame_prefix': ns + '/'  # Añade orion_arm/ a cada link del URDF
            }]
        ),

        Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            name='foxglove_bridge',
             parameters=[{'port': 8765,
                      'send_buffer_limit': 10000000}]   
        ),
        Node(
            package='nixito_perception',
            executable='vision',
            name='vision_node',
            namespace='nixito',
            output='screen',
        ),
        
        Node(
            package = 'nixito_drivers',
            executable = 'serial_data',
            name = 'serial_data',
            output = 'screen',
            
        ),
        Node(
            package = 'orion_arm',
            executable = 'teleop_arm',
            name = 'teleop_arm',
            output = 'screen',
        ),
        
        ]

        
    )