from launch import LaunchDescription
from launch.actions import EmitEvent, ExecuteProcess, RegisterEventHandler
from launch_ros.actions import Node, LifecycleNode
from launch_ros.events.lifecycle import ChangeState
from launch.event_handlers import OnProcessIO, OnProcessExit
from lifecycle_msgs.msg import Transition
import glob
import os
import subprocess


def get_video_device(vendor_id="1bcf", product_id="2284"):
    """Busca el /dev/videoX de captura (no metadata) de la cámara UGREEN 2K."""

    by_id_path = "/dev/v4l/by-id/"
    if os.path.isdir(by_id_path):
        for entry in sorted(os.listdir(by_id_path)):
            if "index0" not in entry:
                continue
            if "sunplus" in entry.lower() or "ugreen" in entry.lower():
                real_dev = os.path.realpath(os.path.join(by_id_path, entry))
                print(f"[get_video_device] Encontrado via by-id: {real_dev}")
                return real_dev

    print("[get_video_device] by-id no funcionó, probando fallback...")

    udevadm_bin = "/usr/bin/udevadm" if os.path.exists("/usr/bin/udevadm") else "udevadm"
    v4l2ctl_bin = "/usr/bin/v4l2-ctl" if os.path.exists("/usr/bin/v4l2-ctl") else "v4l2-ctl"

    for dev in sorted(glob.glob("/dev/video*")):
        try:
            info = subprocess.run(
                [udevadm_bin, "info", "--query=property", f"--name={dev}"],
                capture_output=True, text=True, check=True
            ).stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

        if f"ID_VENDOR_ID={vendor_id}" not in info or f"ID_MODEL_ID={product_id}" not in info:
            continue

        try:
            caps = subprocess.run(
                [v4l2ctl_bin, "-d", dev, "--all"],
                capture_output=True, text=True, check=True
            ).stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

        if "Video Capture" in caps and "Metadata Capture" not in caps:
            print(f"[get_video_device] Encontrado via fallback: {dev}")
            return dev

    raise RuntimeError(f"No se encontró la cámara {vendor_id}:{product_id}")


def generate_launch_description():
    cam_principal = Node(
        package='usb_cam',
        executable="usb_cam_node_exe",
        name="usb_cam",
        namespace='principal',
        output='screen',
        parameters=[{
            'video_device': get_video_device(),
        }]
    )

    realsense = ExecuteProcess(
        cmd=[
            'ros2', 'launch', 'realsense2_camera', 'rs_launch.py',
            'align_depth.enable:=true',
            'rgb_camera.color_profile:=640x480x30',
        ],
        cwd='/home/angel',
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

    return LaunchDescription([
        cam_principal,
        realsense,
        thermal_camera,
        foxglove,
        vision_node,
        vision_maze,
        configure_vision,
        configure_maze,
    ])