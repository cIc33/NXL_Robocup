from launch import LaunchDescription
from launch.actions import EmitEvent, ExecuteProcess, RegisterEventHandler
from launch_ros.actions import Node, LifecycleNode
from launch_ros.events.lifecycle import ChangeState
from launch.event_handlers import OnProcessIO, OnProcessExit
from lifecycle_msgs.msg import Transition
import glob
import os
import subprocess



def generate_launch_description():

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
        realsense,
        thermal_camera,
        foxglove,
        vision_node,
        vision_maze,
        configure_vision,
        configure_maze,
    ])