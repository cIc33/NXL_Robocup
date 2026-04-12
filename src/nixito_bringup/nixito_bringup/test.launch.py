from launch import LaunchDescription
from launch.actions import EmitEvent, ExecuteProcess, RegisterEventHandler
from launch_ros.actions import Node, LifecycleNode
from launch_ros.events.lifecycle import ChangeState
from launch.event_handlers import OnProcessIO
from lifecycle_msgs.msg import Transition
import xacro
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    pkg_name = 'nixito_description'
    urdf_file_name = 'NIXITO 2026.SLDASM.urdf'
    pkg_share = get_package_share_directory(pkg_name)
    urdf_path = os.path.join(pkg_share, 'urdf', urdf_file_name)
    robot_desc = xacro.process_file(urdf_path).toxml()

    # ── GoPro ──────────────────────────────────────────────────────────────
    restart_gopro = ExecuteProcess(
        cmd=['curl', '-s', 'http://172.23.197.51:8080/gp/gpWebcam/STOP'],
        output='screen'
    )

    gopro = ExecuteProcess(
        cmd=['sudo', './gopro', 'webcam', '-a', '-n', '-r', '480', '-i', '172.23.197.51'],
        cwd='/home/aicistemthor/v4l2loopback/gopro_as_webcam_on_linux',  # ✅ fix path
        output='screen'
    )

    ffplay = ExecuteProcess(
        cmd=['ffplay', '/dev/video42'],
        output='screen'
    )

    # ── Cámara brazo ───────────────────────────────────────────────────────
    usb_cam_brazo = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='usb_cam',
        namespace='brazo',
        output='screen',
        parameters=[{
            'video_device': '/dev/video0',   # ✅ era /dev/video0, no existe
            'image_width': 640,
            'image_height': 480,
            'framerate': 30.0,
            'brightness': -1,
            'contrast': -1,
            'saturation': -1,
            'sharpness': -1,
            'autofocus': True,
        }]
    )

    
    thermal = Node(
        package='nixito_perception',
        executable='thermal',
        name='thermal',
        namespace='',
        output='screen'
    )
    # ── Vision lifecycle ───────────────────────────────────────────────────
    vision_node = LifecycleNode(
        package='nixito_perception',
        executable='vision',
        name='vision_node',
        namespace='',
        output='screen'
    )


    configure_vision = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=lambda action: action == vision_node,
            transition_id=Transition.TRANSITION_CONFIGURE,
        )
    )

    # ── Foxglove ───────────────────────────────────────────────────────────
    # Si sigue fallando, corre antes: fuser -k 8765/tcp
    foxglove = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        parameters=[{
            'port': 8765,
            'send_buffer_limit': 10000000
        }]
    )

    # ── Esperar video GoPro antes de lanzar ffplay ─────────────────────────
    ffplay_launched = [False]

    def check_video_ready(event):
        if ffplay_launched[0]:
            return []
        text = event.text.decode() if isinstance(event.text, bytes) else event.text
        if 'video4linux2' in text:
            ffplay_launched[0] = True
            return [ffplay]
        return []

    esperar_video = RegisterEventHandler(
        OnProcessIO(
            target_action=gopro,
            on_stderr=check_video_ready
        )
    )

    return LaunchDescription([
        restart_gopro,
        gopro,
        esperar_video,
        foxglove,
        usb_cam_brazo,
        thermal,
        vision_node,
        configure_vision,
    ])
