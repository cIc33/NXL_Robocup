from launch import LaunchDescription
from launch.actions import EmitEvent
from launch_ros.actions import Node
from launch_ros.actions import LifecycleNode
from launch_ros.events.lifecycle import ChangeState
from launch.actions import ExecuteProcess, EmitEvent, RegisterEventHandler
from lifecycle_msgs.msg import Transition
from launch.event_handlers import OnProcessIO
import xacro
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    restart_gopro = ExecuteProcess(
        cmd=['curl', '-s', 'http://172.23.197.51:8080/gp/gpWebcam/STOP'],
        output='screen'
    )
    
    gopro = ExecuteProcess(
        cmd=['sudo', './gopro', 'webcam', '-a', '-n', '-r', '480', '-i', '172.23.197.51'],
        cwd='/home/aicistemthor/v4l2loopback/gopro_as_webcam_on_linux',
        output='screen'
    )


    ffplay = ExecuteProcess(
        cmd=['ffplay', '/dev/video42'],
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
        namespace='',
        output='screen'
    )

    configure_vision = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=lambda action: action == vision_node,
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

    ffplay_launched = [False]

    def check_video_ready(event):
        if ffplay_launched[0]:
            return[]
        if 'video4linux2' in event.text.decode():
            ffplay_launched[0] = True
            return[ffplay]
        return[]
    
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
        vision_node,
        vision_maze,
        configure_vision,
    ])
