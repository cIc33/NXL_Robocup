from launch import LaunchDescription
from launch.actions import EmitEvent, ExecuteProcess, RegisterEventHandler
from launch_ros.actions import Node, LifecycleNode
from launch_ros.events.lifecycle import ChangeState
from launch.event_handlers import OnProcessIO
from lifecycle_msgs.msg import Transition


def generate_launch_description():

    restart_gopro = ExecuteProcess(
        cmd=['curl', '-s', 'http://172.23.197.51:8080/gp/gpWebcam/STOP'],
        output='screen'
    )

    realsense = ExecuteProcess(
        cmd= ['ros2', 'launch', 'realsense2_camera', 'rs_launch.py', 'align_depth.enable:=true'],
        cwd='/home/angel',
        output='screen'
    )

    gopro = ExecuteProcess(
        cmd=['sudo', './gopro', 'webcam', '-a', '-n', '-r', '480', '-i', '172.23.197.51'],
        cwd='/home/angel/NXL_Robocup/src/nixito_perception/drivers/gopro_as_webcam_on_linux',
        output='screen'
    )

    gopro_camera = Node(
        package='nixito_perception',
        executable='gopro',
        name='gopro_camera',
        output='screen',
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

    cam_trasera = Node(
        package='usb_cam',
        executable="usb_cam_node_exe",
        name="usb_cam",
        namespace='reversa',
        output='screen',
        parameters=[{
        'video_device': '/dev/video8',
        'image_width': 640,
        'image_height': 480,
        'framerate': 30.0,
        'io_method': 'mmap',
        'brightness': -1,
        'contrast': -1,
        'saturation': -1,
        'sharpness': -1,
        'autofocus': True,
        }]
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

    # ── Esperar que ffmpeg confirme que está escribiendo al device ──────
    camera_launched = [False]

    def check_video_ready(event):
        if camera_launched[0]:
            return []
        try:
            text = event.text.decode(errors='ignore')
        except Exception:
            return []
        # Esta línea aparece en stderr de ffmpeg justo cuando empieza
        # a escribir frames a /dev/video42
        if 'video4linux2' in text:
            camera_launched[0] = True
            return [gopro_camera]
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
        realsense,
        cam_trasera,
        thermal_camera,
        foxglove,
        vision_node,
        vision_maze,
        configure_vision,
        configure_maze,
    ])