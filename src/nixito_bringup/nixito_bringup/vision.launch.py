from launch import LaunchDescription
from launch.actions import EmitEvent, ExecuteProcess, RegisterEventHandler
from launch_ros.actions import Node, LifecycleNode
from launch_ros.events.lifecycle import ChangeState
from launch.event_handlers import OnProcessIO, OnProcessExit
from lifecycle_msgs.msg import Transition


def generate_launch_description():
    restart_gopro = ExecuteProcess(
        cmd=['curl', '-s', 'http://172.23.197.51:8080/gp/gpWebcam/STOP'],
        output='screen'
    )

    realsense = ExecuteProcess(
        cmd=[
            'ros2', 'launch', 'realsense2_camera', 'rs_launch.py',
            'align_depth.enable:=true',
            'rgb_camera.color_profile:=640x480x30',
        ],
        cwd='/home/nixito',
        output='screen'
    )

    gopro = ExecuteProcess(
        cmd=['sudo', './gopro', 'webcam', '-a', '-n', '-r', '480', '-i', '172.23.197.51'],
        cwd='/home/nixito/NXL_Robocup/src/nixito_perception/drivers/gopro_as_webcam_on_linux',
        output='screen'
    )

    ffplay = ExecuteProcess(
        cmd=['ffplay', '/dev/video42'],
        output='screen'
    )

    thermal_camera = Node(
        package='nixito_perception',
        executable='thermal',
        name='thermal_topdon',
        namespace='Termica',
        output='screen'
    )

    vision_node = LifecycleNode(
        package='nixito_perception',
        executable='vision',
        name='vision_node',
        namespace='',
        output='screen'
    )

    vision_maze = LifecycleNode(
        package='nixito_perception',
        executable='vision_maze',
        name='vision_maze',
        namespace='maze',
        output='screen'
    )

    configure_vision = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=lambda action: action == vision_node,
            transition_id=Transition.TRANSITION_CONFIGURE,
        )
    )

    configure_maze = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=lambda action: action == vision_maze,
            transition_id=Transition.TRANSITION_CONFIGURE,
        )
    )

    foxglove = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        parameters=[{
            'port': 8765,
            'send_buffer_limit': 10000000
        }]
    )

    # ── Esperar a que el curl STOP termine antes de arrancar el gopro ──
    lanzar_gopro_tras_stop = RegisterEventHandler(
        OnProcessExit(
            target_action=restart_gopro,
            on_exit=[gopro],
        )
    )

    # ── Esperar a que ffmpeg confirme que está escribiendo al device ───
    ffplay_launched = [False]

    def check_video_ready(event):
        if ffplay_launched[0]:
            return []
        try:
            text = event.text.decode(errors='ignore')
        except Exception:
            return []
        # Esta línea aparece en stderr de ffmpeg justo cuando empieza
        # a escribir frames a /dev/video42
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
        lanzar_gopro_tras_stop,
        esperar_video,
        realsense,
        thermal_camera,
        foxglove,
        vision_node,
        vision_maze,
        configure_vision,
        configure_maze,
    ])