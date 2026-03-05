from setuptools import setup
import os
from glob import glob

package_name = 'data_collector'

def get_data_files(directory, install_prefix):
    """递归获取目录下所有文件并生成正确的data_files格式"""
    data_files = []
    
    for root, dirs, files in os.walk(directory):
        # 只处理有文件的目录
        if files:
            # 计算目标安装路径
            rel_path = os.path.relpath(root, directory)
            if rel_path == '.':
                install_path = os.path.join(install_prefix, os.path.basename(directory))
            else:
                install_path = os.path.join(install_prefix, os.path.basename(directory), rel_path)
            
            # 获取该目录下的所有文件（完整路径）
            file_list = [os.path.join(root, f) for f in files]
            data_files.append((install_path, file_list))
    
    return data_files

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        # ROS2 必需文件
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # 安装 worlds 文件
        ('share/' + package_name + '/worlds', 
            glob('worlds/*.sdf') + glob('worlds/*.world')),
        
        # 安装 launch 文件
        ('share/' + package_name + '/launch', 
            glob('launch/*.py')),
        
        # 递归安装 models 目录（包含所有子目录和文件）
        *get_data_files('models', 'share/' + package_name),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='skj',
    maintainer_email='skj@todo.todo',
    description='Dataset collector for pallet using RGBD camera in Gazebo.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dataset_collector = data_collector.dataset_collector:main',
            'random_pallet_pose = data_collector.random_pallet_pose:main',
            'enhanced_random_controller = data_collector.enhanced_random_controller:main',
        ],
    },
)

