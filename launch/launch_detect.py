from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
import launch_ros.actions
from launch.substitutions import LaunchConfiguration
import os
import yaml
from launch.substitutions import EnvironmentVariable
import pathlib
import launch.actions
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    pkg_share = launch_ros.substitutions.FindPackageShare(package="plant_shooter").find(
        "plant_shooter"
    )
            
    return LaunchDescription([
        launch_ros.actions.Node(
            package='plant_shooter',
            executable='Spotter',
            name='Spotter',
            output='screen',
           ),
        launch_ros.actions.Node(
                package="image_transport",
                executable="republish",
                name="republish",
                output="screen",
                arguments=["raw", "in:=/video_frames", "out:=/image_repub"],
                # parameters=[{'in':"video_frames"},{'out':"image_repub"}],
            ), 
    ])


