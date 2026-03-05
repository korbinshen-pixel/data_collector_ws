#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
import cv2
import numpy as np
from datetime import datetime


class DatasetCollector(Node):
    def __init__(self):
        super().__init__('dataset_collector')

        # 要订阅的话题名称（和 SDF 里的插件一致）
        rgb_topic = '/rgbd/rgb_camera/image_raw'
        depth_topic = '/rgbd/depth_camera/depth/image_raw'
        model_states_topic = '/gazebo/model_states'  # gazebo_ros 会提供

        # CVBridge 用于 ROS Image <-> OpenCV 图像转换
        self.bridge = CvBridge()

        # 数据保存根目录
        self.output_root = os.path.expanduser('~/pallet_dataset')
        os.makedirs(self.output_root, exist_ok=True)

        # 为了防止不同运行混在一起，创建时间戳子目录
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(self.output_root, now)
        os.makedirs(self.run_dir, exist_ok=True)

        self.rgb_dir = os.path.join(self.run_dir, 'rgb')
        self.depth_dir = os.path.join(self.run_dir, 'depth')
        self.pose_file = os.path.join(self.run_dir, 'poses.txt')

        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)

        self.get_logger().info(f'Dataset will be saved to: {self.run_dir}')

        # 用于同步：记录最近到达的 RGB / Depth / Pose
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_pallet_pose = None
        self.latest_camera_pose = None
        self.frame_id = 0

        # 订阅话题
        self.rgb_sub = self.create_subscription(
            Image, rgb_topic, self.rgb_callback, 10)

        self.depth_sub = self.create_subscription(
            Image, depth_topic, self.depth_callback, 10)

        self.model_states_sub = self.create_subscription(
            ModelStates, model_states_topic, self.model_states_callback, 10)

        # 设一个定时器，固定频率触发保存（例如 2Hz）
        self.timer_period = 0.5  # 秒
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        # 要跟踪的模型名字
        self.pallet_model_name = 'pallet'
        self.camera_model_name = 'rgbd_camera'  # 相机模型名称

        self.get_logger().info('Dataset collector initialized')

    def rgb_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_rgb = cv_image
        except Exception as e:
            self.get_logger().error(f'RGB cv_bridge error: {e}')

    def depth_callback(self, msg: Image):
        try:
            # 使用 passthrough 直接获取原始数据，不转换编码
            cv_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.latest_depth = cv_depth
        except Exception as e:
            self.get_logger().error(f'Depth cv_bridge error: {e}')

    def model_states_callback(self, msg: ModelStates):
        """获取托盘和相机的世界坐标系位姿"""
        # 获取托盘位姿
        if self.pallet_model_name in msg.name:
            idx = msg.name.index(self.pallet_model_name)
            self.latest_pallet_pose = msg.pose[idx]
        
        # 获取相机位姿
        if self.camera_model_name in msg.name:
            idx = msg.name.index(self.camera_model_name)
            self.latest_camera_pose = msg.pose[idx]

    def quaternion_to_rotation_matrix(self, q):
        """将四元数转换为旋转矩阵（纯NumPy实现）"""
        # q = [x, y, z, w]
        x, y, z, w = q[0], q[1], q[2], q[3]
        
        # 归一化
        norm = np.sqrt(x*x + y*y + z*z + w*w)
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
        
        # 构建旋转矩阵
        R = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        
        return R

    def rotation_matrix_to_euler(self, R):
        """将旋转矩阵转换为欧拉角（ZYX顺序，即roll-pitch-yaw）"""
        # 检查万向锁
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0
        
        return roll, pitch, yaw

    def pose_to_transformation_matrix(self, pose: Pose):
        """将Pose转换为4x4变换矩阵"""
        # 位置
        translation = np.array([
            pose.position.x,
            pose.position.y,
            pose.position.z
        ])
        
        # 四元数转旋转矩阵
        quaternion = [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w
        ]
        rotation_matrix = self.quaternion_to_rotation_matrix(quaternion)
        
        # 构建4x4变换矩阵
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = translation
        
        return T

    def compute_relative_pose(self, pallet_pose: Pose, camera_pose: Pose):
        """计算托盘相对于相机的位姿"""
        # 世界坐标系到相机坐标系的变换
        T_world_to_camera = self.pose_to_transformation_matrix(camera_pose)
        
        # 世界坐标系到托盘坐标系的变换
        T_world_to_pallet = self.pose_to_transformation_matrix(pallet_pose)
        
        # 相机坐标系到托盘坐标系的相对变换
        # T_camera_to_pallet = inv(T_world_to_camera) * T_world_to_pallet
        T_camera_to_world = np.linalg.inv(T_world_to_camera)
        T_camera_to_pallet = T_camera_to_world @ T_world_to_pallet
        
        # 提取位置
        position = T_camera_to_pallet[:3, 3]
        
        # 提取旋转矩阵并转换为欧拉角
        rotation_matrix = T_camera_to_pallet[:3, :3]
        roll, pitch, yaw = self.rotation_matrix_to_euler(rotation_matrix)
        
        return position, roll, pitch, yaw

    def timer_callback(self):
        """定时器回调，保存数据"""
        # 检查所有数据是否就绪
        if self.latest_rgb is None:
            self.get_logger().warn('No RGB image yet.', throttle_duration_sec=5.0)
            return
        if self.latest_depth is None:
            self.get_logger().warn('No Depth image yet.', throttle_duration_sec=5.0)
            return
        if self.latest_pallet_pose is None:
            self.get_logger().warn('No pallet pose yet.', throttle_duration_sec=5.0)
            return
        if self.latest_camera_pose is None:
            self.get_logger().warn('No camera pose yet.', throttle_duration_sec=5.0)
            return

        # 计算相对位姿
        try:
            position, roll, pitch, yaw = self.compute_relative_pose(
                self.latest_pallet_pose, 
                self.latest_camera_pose
            )
        except Exception as e:
            self.get_logger().error(f'Failed to compute relative pose: {e}')
            import traceback
            traceback.print_exc()
            return

        # 保存
        frame_id_str = f'{self.frame_id:06d}'
        rgb_path = os.path.join(self.rgb_dir, f'rgb_{frame_id_str}.png')
        # 改为 png 文件
        depth_path = os.path.join(self.depth_dir, f'depth_{frame_id_str}.png')

        # 保存RGB图像（8位 PNG）
        cv2.imwrite(rgb_path, self.latest_rgb)

        # ====== 这里开始是修改部分：保存深度为 16bit PNG ======
        depth_img = self.latest_depth

        # 如果是 float 型（典型 32FC1，单位米），转换为 mm 的 uint16
        if depth_img.dtype == np.float32 or depth_img.dtype == np.float64:
            # 将 NaN/Inf 先置 0，避免写文件出问题
            depth_img = np.nan_to_num(depth_img, nan=0.0, posinf=0.0, neginf=0.0)
            depth_mm = (depth_img * 1000.0).clip(0, 65535).astype(np.uint16)
        else:
            # 已经是 16 位整型，直接确保是 uint16
            depth_mm = depth_img.astype(np.uint16)

        cv2.imwrite(depth_path, depth_mm)
        # ====== 修改结束 ======

        # 保存相对位姿到文本（追加）
        # 格式：frame_id x y z roll pitch yaw（欧拉角单位：弧度）
        with open(self.pose_file, 'a') as f:
            f.write(
                f'{frame_id_str} '
                f'{position[0]:.6f} {position[1]:.6f} {position[2]:.6f} '
                f'{roll:.6f} {pitch:.6f} {yaw:.6f}\n'
            )

        # 转换为度数用于日志显示
        roll_deg = np.degrees(roll)
        pitch_deg = np.degrees(pitch)
        yaw_deg = np.degrees(yaw)

        self.get_logger().info(
            f'Saved frame {frame_id_str} - '
            f'Pos: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}), '
            f'Rot: ({roll_deg:.1f}°, {pitch_deg:.1f}°, {yaw_deg:.1f}°)'
        )
        self.frame_id += 1


def main(args=None):
    rclpy.init(args=args)
    node = DatasetCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()





