from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    image_width = LaunchConfiguration('image_width')
    image_height = LaunchConfiguration('image_height')
    video_device = LaunchConfiguration('video_device')

    declare_image_width = DeclareLaunchArgument(
        'image_width', default_value='160',
        description='Width of the camera image'
    )

    declare_image_height = DeclareLaunchArgument(
        'image_height', default_value='120',
        description='Height of the camera image'
    )

    declare_video_device = DeclareLaunchArgument(
        'video_device', default_value='/dev/video0',
        description='Video device for USB camera'
    )

    usb_cam_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='usb_cam',
        output='screen',
        parameters=[
            {'image_width': image_width},
            {'image_height': image_height},
            {'video_device': video_device},
            # aquí podries afegir frame_id, framerate, etc.
            # {'camera_frame_id': 'usb_cam_link'},
            # {'framerate': 30.0},
        ]
    )

    return LaunchDescription([
        declare_image_width,
        declare_image_height,
        declare_video_device,
        usb_cam_node,
    ])
