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

    pkg_name = 'nixito_description'
    urdf_file_name = 'NIXITO 2026.SLDASM.urdf'

    pkg_share = get_package_share_directory(pkg_name)
    params_file = os.path.join(pkg_share, 'config', 'arm_presets.yaml')

    urdf_path = os.path.join(
        pkg_share,
        'urdf',
        urdf_file_name
    )

    robot_description_config = xacro.process_file(urdf_path)
    robot_desc = robot_description_config.toxml()

    restart_gopro = ExecuteProcess(
        cmd=['curl', '-s', 'http://172.23.197.51:8080/gp/gpWebcam/STOP'],
        output='screen'
    )
    
    gopro = ExecuteProcess(
        cmd=['sudo','gopro','webcam', '-a', '-n','-r','480'],
        cwd='/home/sahid/v4l2loopback/gopro_as_webcam_on_linux',
        output='screen'
    )


    ffplay = ExecuteProcess(
        cmd=['ffplay', '/dev/video42'],
        output='screen'
    )

    usb_cam_brazo = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='usb_cam',
        namespace='brazo',
        output='screen',
        parameters=[{
            'video_device': '/dev/video0',
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

    thermal = Node(
        package='nixito_perception',
        executable='thermal',
        name='thermal_cam',
        output='screen'
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

    teleopArm = Node(
        package='orion_arm',
        executable='teleop_arm',
        name='teleop_arm',
        output='screen'
    )

    driverArm = Node(
        package='orion_arm',
        executable='orion_driver',
        name='orion_driver',
        output='screen'
    )

    robotState = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_desc,
        }],
        remappings=[
            ('/joint_states', '/nixito/joint_states'),
        ]
    ) 

    armState = Node(
        package='orion_arm',
        executable='bridge_esp32',
        name='bridge_esp32',
        output='screen',
    )

    serial_data = Node(
        package='nixito_drivers',
        executable='serial_data',
        name='serial_data',
        output='screen',
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
        usb_cam_brazo,
        vision_node,
        configure_vision,
        thermal,
        teleopArm,
        driverArm,
        robotState,
        armState,
        serial_data,
    ])