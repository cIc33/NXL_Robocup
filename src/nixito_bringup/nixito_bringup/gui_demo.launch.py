from launch import LaunchDescription
from launch_ros.actions import LifecycleNode
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, TimerAction, RegisterEventHandler
from launch.event_handlers import OnProcessStart

def generate_launch_description():

    # 1. Nodo de Cámara: Corregido a /dev/video0 y eliminando basura de logs
    usb_cam_node = Node(
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
    )

    # 2. Nodo de Visión: Declarado como LifecycleNode nativo
    vision_node = LifecycleNode(
        package="nixito_perception",
        executable="vision",
        name="vision_node",
        namespace="",
        output="screen",
        # Asegúrate de que el script cargue el .engine que generaste manualmente
    )

    # 3. GUI y Bridge
    gui_node = Node(
        package="nixito_gui",
        executable="gui_demo",
        name="gui_demo_node",
        output="screen",
    )

    foxglove_bridge = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        parameters=[{'port': 8765,
                     'send_buffer_limit': 10000000}]
    )
    
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', '/home/axelcg_7905/nixito_ws/nixito_gui/config/nixito_embedded.rviz']
    )

    # 4. Automatización del Ciclo de Vida: Forma PRO
    # En lugar de un Timer fijo, esperamos a que el proceso realmente inicie
    configure_vision = ExecuteProcess(
        cmd=["ros2", "lifecycle", "set", "/vision_node", "configure"],
        output="screen",
    )


 

    return LaunchDescription([
        usb_cam_node,
        vision_node,
        gui_node,
        foxglove_bridge,
        configure_vision,
        rviz
    ])