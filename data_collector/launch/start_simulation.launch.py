from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('data_collector')

    # 世界文件路径（安装后在 share/data_collector/worlds 里）
    world_path = os.path.join(pkg_share, 'worlds', 'warehouse_world.sdf')

    # 启动 Gazebo11 + 世界
    gazebo = ExecuteProcess(
        cmd=[
            'gazebo',
            world_path,
            '--verbose'
        ],
        output='screen'
    )

    # 启动数据采集节点
    collector_node = Node(
        package='data_collector',
        executable='dataset_collector',
        name='dataset_collector',
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        collector_node
    ])

