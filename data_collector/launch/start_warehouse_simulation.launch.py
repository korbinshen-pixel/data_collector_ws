from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('data_collector')

    # 仓库世界文件
    world_path = os.path.join(pkg_share, 'worlds', 'warehouse_world.sdf')

    # 启动 Gazebo11
    gazebo = ExecuteProcess(
        cmd=['gazebo', world_path, '--verbose'],
        output='screen'
    )

    # 启动 gazebo_ros 节点（提供 spawn/delete 服务）
    gazebo_ros_node = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[],
        output='screen',
        parameters=[{'use_sim_time': True}],
        # 这个节点会提供 /spawn_entity 和 /delete_entity 服务
        name='gazebo_spawner'
    )

    # 启动数据采集节点
    collector_node = Node(
        package='data_collector',
        executable='dataset_collector',
        name='dataset_collector',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # 启动增强随机控制节点
    random_controller = Node(
        package='data_collector',
        executable='enhanced_random_controller',
        name='enhanced_random_controller',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    return LaunchDescription([
        gazebo,
        gazebo_ros_node,
        collector_node,
        random_controller
    ])
