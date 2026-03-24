#!/usr/bin/env python3
"""
dataset_collector.py - Gazebo ROS2 数据采集节点
=================================================
直接采集并保存为 EfficientPose 所需的 Linemod 格式数据集。

输出目录结构:
  ~/pallet_dataset/
  ├── data/
  │   └── 01/
  │       ├── rgb/           0000.png, 0001.png, ...
  │       ├── mask/          0000.png, 0001.png, ... (需后续生成或用分割话题)
  │       ├── depth/         0000.png, 0001.png, ... (16bit, 单位mm)
  │       ├── gt.yml         位姿标注 (旋转矩阵 + 平移, 单位mm)
  │       ├── info.yml       每帧相机内参
  │       ├── train.txt      训练集ID列表
  │       └── test.txt       测试集ID列表
  └── models/
      └── models_info.yml    模型尺寸信息 (手动或自动生成)

用法:
  ros2 run your_package dataset_collector

采集完成后按 Ctrl+C, 节点会自动:
  1. 写出 gt.yml / info.yml
  2. 按比例划分 train.txt / test.txt
  3. 生成 models/models_info.yml
  4. 从位姿投影生成 mask/ 图像 (若无分割话题)
"""

import os
import sys
import random
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml
from datetime import datetime


class DatasetCollector(Node):
    def __init__(self):
        super().__init__('dataset_collector')

        # ===================== 可配置参数 =====================
        # ROS话题
        rgb_topic = '/rgbd/rgb_camera/image_raw'
        depth_topic = '/rgbd/depth_camera/depth/image_raw'
        model_states_topic = '/gazebo/model_states'

        # 模型名称 (需与Gazebo模型名一致)
        self.pallet_model_name = 'pallet'
        self.camera_model_name = 'rgbd_camera'

        # 物体ID (Linemod格式使用两位数字)
        self.object_id = 1

        # 相机内参 (根据你的SDF/URDF中的相机参数填写)
        # 如果订阅了CameraInfo话题则会自动更新
        self.fx = 554.25
        self.fy = 554.25
        self.cx = 320.0
        self.cy = 240.0

        # ★ 托盘尺寸随机范围 (米), 每帧独立采样
        # 格式: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        self.pallet_size_range = (
            (0.5, 0.7),   # x (长) 范围
            (0.5, 0.7),   # y (宽) 范围
            (0.12, 0.18), # z (高) 范围
        )

        # 采集频率 (Hz)
        self.collect_hz = 2.0

        # 训练/测试集比例
        self.train_ratio = 0.80

        

        # ===================== 输出目录 =====================
        self.output_root = os.path.expanduser('~/pallet_dataset')
        
        # ★ 用时间戳创建独立的数据收集目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.obj_dir = os.path.join(
            self.output_root, 
            'data', 
            f'{timestamp}_{self.object_id:02d}'  # 例如: 20260305_184215_01
        )
        self.model_dir = os.path.join(self.output_root, 'models')

        self.rgb_dir = os.path.join(self.obj_dir, 'rgb')
        self.mask_dir = os.path.join(self.obj_dir, 'mask')
        self.depth_dir = os.path.join(self.obj_dir, 'depth')

        for d in [self.rgb_dir, self.mask_dir, self.depth_dir, self.model_dir]:
            os.makedirs(d, exist_ok=True)

        self.get_logger().info(f'Dataset output: {self.output_root}')
        self.get_logger().info(f'Collection dir: {self.obj_dir}')  # ★ 新增日志

        # ===================== 内部状态 =====================
        self.bridge = CvBridge()
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_pallet_pose = None
        self.latest_camera_pose = None
        self.frame_id = 0

        # 累积数据 (用于最终写出 yaml)
        self.gt_dict = {}      # {frame_id: [{cam_R_m2c, cam_t_m2c, obj_id}]}
        self.info_dict = {}    # {frame_id: {cam_K: [9 floats]}}

        # ===================== 订阅 =====================
        self.rgb_sub = self.create_subscription(
            Image, rgb_topic, self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, depth_topic, self.depth_callback, 10)
        self.model_states_sub = self.create_subscription(
            ModelStates, model_states_topic, self.model_states_callback, 10)


        # 定时采集
        self.timer = self.create_timer(1.0 / self.collect_hz, self.timer_callback)

    # ===================== 回调函数 =====================

    def rgb_callback(self, msg: Image):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'RGB cv_bridge error: {e}')

    def depth_callback(self, msg: Image):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Depth cv_bridge error: {e}')

    def model_states_callback(self, msg: ModelStates):
        if self.pallet_model_name in msg.name:
            idx = msg.name.index(self.pallet_model_name)
            self.latest_pallet_pose = msg.pose[idx]
        if self.camera_model_name in msg.name:
            idx = msg.name.index(self.camera_model_name)
            self.latest_camera_pose = msg.pose[idx]


    # ===================== 坐标变换工具 =====================

    def quaternion_to_rotation_matrix(self, qx, qy, qz, qw):
        """四元数 -> 3x3旋转矩阵"""
        norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
        return np.array([
            [1 - 2*(qy*qy + qz*qz),   2*(qx*qy - qw*qz),   2*(qx*qz + qw*qy)],
            [  2*(qx*qy + qw*qz),   1 - 2*(qx*qx + qz*qz),   2*(qy*qz - qw*qx)],
            [  2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),   1 - 2*(qx*qx + qy*qy)]
        ])

    def pose_to_transform(self, pose: Pose):
        """ROS Pose -> 4x4齐次变换矩阵"""
        R = self.quaternion_to_rotation_matrix(
            pose.orientation.x, pose.orientation.y,
            pose.orientation.z, pose.orientation.w)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
        return T

    def compute_relative_pose(self, pallet_pose: Pose, camera_pose: Pose):
        """
        计算托盘相对于相机 **光学坐标系** 的位姿 (cam_R_m2c, cam_t_m2c)

        关键: Gazebo model_states 给出的是 body frame (X前, Y左, Z上)
            EfficientPose/OpenCV 需要 optical frame (X右, Y下, Z前)
            两者之间需要乘一个固定旋转矩阵
        """
        T_w2c_body = self.pose_to_transform(camera_pose)  # world → camera body
        T_w2m = self.pose_to_transform(pallet_pose)        # world → model

        # ★ Gazebo body frame → OpenCV optical frame 的变换
        # 等价于 URDF 中 rpy="-pi/2, 0, -pi/2"
        # body:   X前, Y左, Z上
        # optical: X右, Y下, Z前
        #
        #  optical_X =  body_(-Y)  → [0, -1, 0]
        #  optical_Y =  body_(-Z)  → [0,  0, -1]
        #  optical_Z =  body_(+X)  → [1,  0, 0]
        R_body_to_optical = np.array([
            [ 0, -1,  0],
            [ 0,  0, -1],
            [ 1,  0,  0]
        ], dtype=np.float64)

        T_body_to_optical = np.eye(4)
        T_body_to_optical[:3, :3] = R_body_to_optical

        # world → camera optical frame
        T_w2c_optical = T_body_to_optical @ np.linalg.inv(T_w2c_body)

        # model in camera optical frame
        T_c2m = T_w2c_optical @ T_w2m

        R_m2c = T_c2m[:3, :3]
        t_m2c = T_c2m[:3, 3] * 1000.0  # 米 → 毫米

        return R_m2c, t_m2c


    # ===================== ★ 随机尺寸采样 =====================

    def sample_pallet_size(self):
        """在配置的范围内随机采样一个托盘尺寸 (米)，并返回对应的26个包围盒点 (mm)"""
        sx_m = random.uniform(*self.pallet_size_range[0])
        sy_m = random.uniform(*self.pallet_size_range[1])
        sz_m = random.uniform(*self.pallet_size_range[2])

        sx, sy, sz = sx_m * 1000.0, sy_m * 1000.0, sz_m * 1000.0

        corners = np.array([
            [-sx/2, -sy/2, -sz/2], [ sx/2, -sy/2, -sz/2],
            [ sx/2,  sy/2, -sz/2], [-sx/2,  sy/2, -sz/2],
            [-sx/2, -sy/2,  sz/2], [ sx/2, -sy/2,  sz/2],
            [ sx/2,  sy/2,  sz/2], [-sx/2,  sy/2,  sz/2],
        ], dtype=np.float64)

        edge_midpoints = np.array([
            [0, -sy/2, -sz/2], [sx/2, 0, -sz/2], [0, sy/2, -sz/2], [-sx/2, 0, -sz/2],
            [0, -sy/2, sz/2], [sx/2, 0, sz/2], [0, sy/2, sz/2], [-sx/2, 0, sz/2],
            [-sx/2, -sy/2, 0], [sx/2, -sy/2, 0], [sx/2, sy/2, 0], [-sx/2, sy/2, 0],
        ], dtype=np.float64)

        face_centers = np.array([
            [0, 0, -sz/2], [0, 0, sz/2],
            [0, -sy/2, 0], [0, sy/2, 0],
            [-sx/2, 0, 0], [sx/2, 0, 0],
        ], dtype=np.float64)

        bbox_corners_mm = np.vstack([corners, edge_midpoints, face_centers])
        return (sx_m, sy_m, sz_m), bbox_corners_mm


    # ===================== Mask生成 =====================

    def generate_mask_from_pose(self, img_shape, R_m2c, t_m2c, bbox_corners_mm):
        """
        从位姿投影3D包围盒角点, 取凸包作为分割mask
        (简易方法, 适用于凸物体; 更精确需要渲染器)

        Args:
            img_shape: (H, W, C)
            R_m2c: (3,3) 旋转矩阵
            t_m2c: (3,) 平移向量(mm)
            bbox_corners_mm: (26, 3) 当前帧的包围盒点
        Returns:
            mask: (H, W, 3) uint8, 物体区域为白色255
        """
        K = np.array([[self.fx, 0, self.cx],
                      [0, self.fy, self.cy],
                      [0, 0, 1]], dtype=np.float64)

        # 3D -> 相机坐标
        pts_cam = R_m2c @ bbox_corners_mm.T + t_m2c.reshape(3, 1)

        # 过滤掉在相机后面的点
        if np.any(pts_cam[2, :] <= 0):
            return np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)

        # 投影到2D
        pts_2d = K @ pts_cam
        pts_2d = (pts_2d[:2, :] / pts_2d[2:3, :]).T  # (8, 2)

        # 取凸包
        hull = cv2.convexHull(pts_2d.astype(np.float32)).astype(np.int32)
        mask = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, (255, 255, 255))

        return mask


    # ===================== 定时采集回调 =====================

    def timer_callback(self):
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

        # ---- 1. 计算相对位姿 (Linemod格式: 旋转矩阵 + mm平移) ----
        try:
            R_m2c, t_m2c = self.compute_relative_pose(
                self.latest_pallet_pose, self.latest_camera_pose)
        except Exception as e:
            self.get_logger().error(f'Pose computation error: {e}')
            return

        # ---- ★ 1b. 随机采样本帧托盘尺寸 ----
        pallet_size_m, bbox_corners_mm = self.sample_pallet_size()

        # ---- 2. 文件名: 4位数字, 无前缀 ----
        fid = self.frame_id
        fname = f'{fid:04d}.png'

        # ---- 3. 保存RGB图像 ----
        cv2.imwrite(os.path.join(self.rgb_dir, fname), self.latest_rgb)

        # ---- 4. 保存深度图 (16bit PNG, 单位mm) ----
        depth_img = self.latest_depth
        if depth_img.dtype == np.float32 or depth_img.dtype == np.float64:
            depth_img = np.nan_to_num(depth_img, nan=0.0, posinf=0.0, neginf=0.0)
            depth_mm = (depth_img * 1000.0).clip(0, 65535).astype(np.uint16)
        else:
            depth_mm = depth_img.astype(np.uint16)
        cv2.imwrite(os.path.join(self.depth_dir, fname), depth_mm)

        # ---- 5. 生成并保存mask (使用本帧尺寸) ----
        mask = self.generate_mask_from_pose(self.latest_rgb.shape, R_m2c, t_m2c, bbox_corners_mm)
        cv2.imwrite(os.path.join(self.mask_dir, fname), mask)


        # ---- 6. 累积gt标注 (Linemod gt.yml格式) ----
        # ★ 新增 pallet_size_m 字段记录本帧实际尺寸
        self.gt_dict[fid] = [{
            'cam_R_m2c': R_m2c.flatten().tolist(),
            'cam_t_m2c': t_m2c.tolist(),
            'obj_id': self.object_id,
            'pallet_size_m': list(pallet_size_m),  # ★ (sx, sy, sz) in meters
        }]

        # ---- 7. 累积相机内参 (Linemod info.yml格式) ----
        # cam_K: 3x3内参矩阵展平为9个float
        cam_K = [self.fx, 0.0, self.cx,
                 0.0, self.fy, self.cy,
                 0.0, 0.0, 1.0]
        self.info_dict[fid] = {
            'cam_K': cam_K
        }

        # ---- 日志 ----
        t_mm = t_m2c
        self.get_logger().info(
            f'Frame {fid:04d} | '
            f't=({t_mm[0]:.1f}, {t_mm[1]:.1f}, {t_mm[2]:.1f})mm | '
            f'size=({pallet_size_m[0]:.3f},{pallet_size_m[1]:.3f},{pallet_size_m[2]:.3f})m | '
            f'Total: {fid + 1} frames')

        self.frame_id += 1

    # ===================== 数据集收尾: 写出YAML和分割文件 =====================

    def finalize_dataset(self):
        """
        在节点关闭时调用, 写出:
          - gt.yml
          - info.yml
          - train.txt / test.txt
          - models/models_info.yml
        """
        n_frames = self.frame_id
        if n_frames == 0:
            self.get_logger().warn('No frames collected, nothing to save.')
            return

        self.get_logger().info(f'Finalizing dataset ({n_frames} frames)...')

        # ---- gt.yml ----
        gt_path = os.path.join(self.obj_dir, 'gt.yml')
        with open(gt_path, 'w') as f:
            yaml.dump(self.gt_dict, f, default_flow_style=False)
        self.get_logger().info(f'  Written: {gt_path}')

        # ---- info.yml ----
        info_path = os.path.join(self.obj_dir, 'info.yml')
        with open(info_path, 'w') as f:
            yaml.dump(self.info_dict, f, default_flow_style=False)
        self.get_logger().info(f'  Written: {info_path}')

        # ---- train.txt / test.txt ----
        all_ids = list(range(n_frames))
        random.shuffle(all_ids)
        n_train = int(n_frames * self.train_ratio)
        train_ids = sorted(all_ids[:n_train])
        test_ids = sorted(all_ids[n_train:])

        train_path = os.path.join(self.obj_dir, 'train.txt')
        with open(train_path, 'w') as f:
            for i in train_ids:
                f.write(f'{i:04d}\n')

        test_path = os.path.join(self.obj_dir, 'test.txt')
        with open(test_path, 'w') as f:
            for i in test_ids:
                f.write(f'{i:04d}\n')

        self.get_logger().info(
            f'  Train: {len(train_ids)} | Test: {len(test_ids)} '
            f'({self.train_ratio*100:.0f}%/{(1-self.train_ratio)*100:.0f}%)')

        # ---- models/models_info.yml ----
        # ★ 使用尺寸范围的中值作为代表尺寸
        sx = (self.pallet_size_range[0][0] + self.pallet_size_range[0][1]) / 2 * 1000.0
        sy = (self.pallet_size_range[1][0] + self.pallet_size_range[1][1]) / 2 * 1000.0
        sz = (self.pallet_size_range[2][0] + self.pallet_size_range[2][1]) / 2 * 1000.0
        diameter = float(np.sqrt(sx**2 + sy**2 + sz**2))
        models_info = {
            self.object_id: {
                'diameter': diameter,
                'min_x': float(-sx / 2),
                'min_y': float(-sy / 2),
                'min_z': float(-sz / 2),
                'size_x': float(sx),
                'size_y': float(sy),
                'size_z': float(sz),
                # ★ 同时记录实际随机范围，供训练脚本参考
                'size_x_range': list(self.pallet_size_range[0]),
                'size_y_range': list(self.pallet_size_range[1]),
                'size_z_range': list(self.pallet_size_range[2]),
            }
        }
        models_info_path = os.path.join(self.model_dir, 'models_info.yml')
        with open(models_info_path, 'w') as f:
            yaml.dump(models_info, f, default_flow_style=False)
        self.get_logger().info(f'  Written: {models_info_path}')
        self.get_logger().info(f'  Pallet diameter (mean): {diameter:.1f} mm')

        self.get_logger().info('='*50)
        self.get_logger().info(f'Dataset complete! {n_frames} frames saved to:')
        self.get_logger().info(f'  {self.output_root}')
        self.get_logger().info('='*50)
        self.get_logger().info('')
        self.get_logger().info('Next steps:')
        self.get_logger().info('  1. (可选) 准备 obj_01.ply 3D模型放入 models/ 目录')
        self.get_logger().info('  2. 运行训练:')
        self.get_logger().info(f'     python train_pallet.py --dataset-type pallet \\')
        self.get_logger().info(f'         --dataset-path {self.output_root} \\')
        self.get_logger().info(f'         --object-id {self.object_id} --phi 0 --weights imagenet')


def main(args=None):
    rclpy.init(args=args)
    node = DatasetCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 关键: 在退出时写出所有yaml和分割文件
        node.finalize_dataset()
        node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
