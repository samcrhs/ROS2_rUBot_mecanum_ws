from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    mecanum_serial_port = LaunchConfiguration('mecanum_serial_port')
    baud_rate = LaunchConfiguration('baud_rate')
    loop_rate = LaunchConfiguration('loop_rate')
    encoder_cpr = LaunchConfiguration('encoder_cpr')

    declare_mecanum_serial_port = DeclareLaunchArgument(
        'mecanum_serial_port', default_value='/dev/ttyACM0',
        description='Serial port for Nano mecanum driver'
    )

    declare_baud_rate = DeclareLaunchArgument(
        'baud_rate', default_value='57600',
        description='Baud rate for Nano mecanum driver'
    )

    declare_loop_rate = DeclareLaunchArgument(
        'loop_rate', default_value='30',
        description='Loop rate (Hz) for nano driver node'
    )

    declare_encoder_cpr = DeclareLaunchArgument(
        'encoder_cpr', default_value='1320',
        description='Encoder counts per revolution'
    )

    # Incloure el launch “oficial” del driver (del package my_robot_driver)
    nano_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('my_robot_driver'),
                'launch',
                'rubot_nano_driver_mecanum.launch.py'
            )
        ),
        launch_arguments={
            'serial_port': mecanum_serial_port,
            'baud_rate': baud_rate,
            'loop_rate': loop_rate,
            'encoder_cpr': encoder_cpr,
        }.items()
    )

    return LaunchDescription([
        declare_mecanum_serial_port,
        declare_baud_rate,
        declare_loop_rate,
        declare_encoder_cpr,
        nano_launch,
    ])
