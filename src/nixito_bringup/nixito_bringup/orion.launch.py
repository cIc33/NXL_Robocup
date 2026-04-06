import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import xacro

def generate_launch_description():

    # 1. Configuración de rutas y variables
    pkg_name = 'nixito_description' 
    urdf_file_name = 'orion_arm.SLDASM.urdf'  
    ns = 'orion_arm' # Namespace unificado
    
    pkg_share = get_package_share_directory(pkg_name)
    params_file = os.path.join(pkg_share, 'config', 'arm_presets.yaml')
    
    urdf_path = os.path.join(
        get_package_share_directory(pkg_name),
        'urdf',
        urdf_file_name)

    # 2. Procesar URDF/Xacro
    robot_description_config = xacro.process_file(urdf_path)
    robot_desc = robot_description_config.toxml()

    return LaunchDescription([
        
        # Publicador de estado del robot (TF)
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

        # Bridge ESP32
        Node(
            package='orion_arm',
            executable='bridge_esp32',
            name='bridge_esp32',
            namespace=ns,
            output='screen'
        ),

        # Cinemática Inversa y Solvers (Ahora dentro del namespace)
        Node(
            package='orion_arm',
            executable='orion_ik_node',
            name='orion_ik_node',
            output='screen'
        ),
        
        # Driver del brazo
        Node(
            package='orion_arm',
            executable='orion_driver',
            name='orion_driver',
            output='screen'
        ),

        # GUI de control (nixito_gui)



        
        #Nodo teleoperacion brazo
        Node(
            package='orion_arm',
            executable ='teleop_arm',
            name='teleop_arm',
            output='screen',
        ),

     
           
       
    ])