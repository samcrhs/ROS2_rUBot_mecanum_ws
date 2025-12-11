from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    rplidar_serial_port = LaunchConfiguration('rplidar_serial_port')
    rplidar_frame_id = LaunchConfiguration('rplidar_frame_id')

    declare_rplidar_serial_port = DeclareLaunchArgument(
        'rplidar_serial_port', default_value='/dev/ttyUSB0',
        description='Serial port for RPLidar'
    )

    declare_rplidar_frame_id = DeclareLaunchArgument(
        'rplidar_frame_id', default_value='base_link',
        description='Frame ID for RPLidar data'
    )

    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('rplidar_ros'),
                'launch',
                'rplidar_a1_launch.py'
            )
        ),
        launch_arguments={
            'serial_port': rplidar_serial_port,
            'frame_id': rplidar_frame_id,
        }.items()
    )

    return LaunchDescription([
        declare_rplidar_serial_port,
        declare_rplidar_frame_id,
        lidar_launch,
    ])
