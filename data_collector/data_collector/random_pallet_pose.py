#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState, GetEntityState
from gazebo_msgs.msg import EntityState, ModelStates
from geometry_msgs.msg import Pose
import random
import math
import numpy as np


class RandomPalletPose(Node):
    def __init__(self):
        super().__init__('random_pallet_pose')

        # 创建服务客户端
        self.set_state_cli = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        
        # 等待服务可用
        while not self.set_state_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /gazebo/set_entity_state service...')

        # 订阅model_states来获取实时位姿
        self.model_states_sub = self.create_subscription(
            ModelStates,
            '/gazebo/model_states',
            self.model_states_callback,
            10
        )
        
        self.current_camera_actual_pose = None
        self.available_models = []
        self.models_printed = False

        # 模型名称 - 可能需要调整
        self.pallet_name = 'pallet'
        self.camera_name = 'rgbd_camera'  # 这个名称可能不对
        
        # 相机位置范围
        self.camera_x_range = (2.0, 4.0)
        self.camera_y_range = (-1.0, 1.0)
        self.camera_z_range = (0.2, 0.35)
        
        # 相机yaw角度范围
        self.camera_yaw_min = math.radians(90)
        self.camera_yaw_max = math.radians(270)
        
        # 相机参数
        self.camera_fov = 60.0
        
        # 托盘生成参数
        self.pallet_distance_min = 1.5
        self.pallet_distance_max = 3.5
        self.pallet_max_angle = math.radians(60)  # 托盘与相机之间的最大夹角（60度）
        
        # 仓库边界
        self.warehouse_x_range = (-3.5, 3.5)
        self.warehouse_y_range = (-3.5, 3.5)
        
        # 状态控制
        self.pallets_per_camera = 100
        self.pallet_interval = 0.5
        
        # 相机验证参数
        self.camera_position_tolerance = 0.1
        self.camera_orientation_tolerance = 0.1
        self.max_camera_retries = 10
        
        # 目标相机位姿
        self.target_camera_x = None
        self.target_camera_y = None
        self.target_camera_z = None
        self.target_camera_yaw = None
        
        # 当前状态
        self.pallet_count = 0
        self.camera_retry_count = 0
        self.state = 'WAIT_FOR_MODELS'  # 先等待获取模型列表
        self.wait_counter = 0
        
        # 定时器
        self.timer = self.create_timer(self.pallet_interval, self.timer_callback)
        
        self.get_logger().info('Random camera and pallet pose node started.')
        self.get_logger().info('Waiting to detect available models...')

    def model_states_callback(self, msg: ModelStates):
        """获取相机当前位姿"""
        # 保存可用模型列表
        if not self.models_printed:
            self.available_models = list(msg.name)
            self.get_logger().info(f'Available models in Gazebo: {self.available_models}')
            self.models_printed = True
            
            # 尝试找到相机模型
            camera_candidates = [name for name in self.available_models if 'camera' in name.lower() or 'rgbd' in name.lower()]
            if camera_candidates:
                self.get_logger().info(f'Possible camera models: {camera_candidates}')
                # 如果找到候选，使用第一个
                if self.camera_name not in self.available_models and camera_candidates:
                    old_name = self.camera_name
                    self.camera_name = camera_candidates[0]
                    self.get_logger().warn(f'Camera model name changed from "{old_name}" to "{self.camera_name}"')
        
        # 获取相机位姿
        if self.camera_name in msg.name:
            idx = msg.name.index(self.camera_name)
            self.current_camera_actual_pose = msg.pose[idx]
        else:
            # 相机模型不存在，打印警告
            if self.wait_counter % 10 == 0:  # 每10次打印一次，避免刷屏
                self.get_logger().warn(f'Camera model "{self.camera_name}" not found in model_states!')

    def set_new_camera_position(self):
        """设置新的随机相机位置"""
        # 检查相机模型是否存在
        if self.camera_name not in self.available_models:
            self.get_logger().error(f'Cannot set camera position: model "{self.camera_name}" does not exist!')
            self.get_logger().error(f'Available models: {self.available_models}')
            return False
        
        # 随机生成目标相机位置和朝向
        self.target_camera_x = random.uniform(self.camera_x_range[0], self.camera_x_range[1])
        self.target_camera_y = random.uniform(self.camera_y_range[0], self.camera_y_range[1])
        self.target_camera_z = random.uniform(self.camera_z_range[0], self.camera_z_range[1])
        self.target_camera_yaw = random.uniform(self.camera_yaw_min, self.camera_yaw_max)
        
        # 发送设置命令
        success = self.set_entity_pose(
            self.camera_name,
            self.target_camera_x, 
            self.target_camera_y, 
            self.target_camera_z,
            0.0, 0.0, self.target_camera_yaw
        )
        
        if not success:
            self.get_logger().error('Failed to send camera pose command!')
            return False
        
        # 重置状态
        self.pallet_count = 0
        self.camera_retry_count = 0
        
        self.get_logger().info(
            f'\n========== SETTING NEW CAMERA POSITION ==========\n'
            f'Model: "{self.camera_name}"\n'
            f'Target: ({self.target_camera_x:.2f}, {self.target_camera_y:.2f}, {self.target_camera_z:.2f}), '
            f'yaw={math.degrees(self.target_camera_yaw):.1f}°'
        )
        return True

    def verify_camera_position(self):
        """验证相机是否到达目标位置"""
        if self.current_camera_actual_pose is None:
            self.get_logger().warn('Camera pose not available yet from model_states')
            return False
        
        # 获取当前相机实际位姿
        actual_x = self.current_camera_actual_pose.position.x
        actual_y = self.current_camera_actual_pose.position.y
        actual_z = self.current_camera_actual_pose.position.z
        
        # 将当前四元数转换为yaw角
        actual_yaw = self.quaternion_to_yaw(
            self.current_camera_actual_pose.orientation.x,
            self.current_camera_actual_pose.orientation.y,
            self.current_camera_actual_pose.orientation.z,
            self.current_camera_actual_pose.orientation.w
        )
        
        # 计算位置误差
        position_error = math.sqrt(
            (actual_x - self.target_camera_x)**2 +
            (actual_y - self.target_camera_y)**2 +
            (actual_z - self.target_camera_z)**2
        )
        
        # 计算yaw角误差
        yaw_error = abs(self.angle_diff(actual_yaw, self.target_camera_yaw))
        
        self.get_logger().info(
            f'Camera verification:\n'
            f'  Target: ({self.target_camera_x:.3f}, {self.target_camera_y:.3f}, {self.target_camera_z:.3f}), yaw={math.degrees(self.target_camera_yaw):.2f}°\n'
            f'  Actual: ({actual_x:.3f}, {actual_y:.3f}, {actual_z:.3f}), yaw={math.degrees(actual_yaw):.2f}°\n'
            f'  Error: pos={position_error:.4f}m, yaw={math.degrees(yaw_error):.2f}°'
        )
        
        # 检查是否在容限内
        if position_error < self.camera_position_tolerance and yaw_error < self.camera_orientation_tolerance:
            self.get_logger().info('✓ Camera position verified! Starting pallet generation...')
            return True
        else:
            self.get_logger().warn(
                f'✗ Camera position NOT verified '
                f'(pos: {position_error:.4f}m, yaw: {math.degrees(yaw_error):.2f}°)'
            )
            return False

    def timer_callback(self):
        """定时器回调"""
        self.wait_counter += 1
        
        if self.state == 'WAIT_FOR_MODELS':
            # 等待获取模型列表
            if self.models_printed:
                self.get_logger().info('Models detected. Starting camera positioning...')
                self.state = 'SET_CAMERA'
            elif self.wait_counter > 20:
                self.get_logger().error('Timeout waiting for model_states. Check if Gazebo is running properly.')
                self.state = 'SET_CAMERA'  # 尝试继续
            
        elif self.state == 'SET_CAMERA':
            # 设置新相机位置
            if self.set_new_camera_position():
                self.state = 'VERIFY_CAMERA'
            else:
                self.get_logger().error('Failed to set camera. Retrying...')
            
        elif self.state == 'VERIFY_CAMERA':
            # 验证相机位置
            if self.verify_camera_position():
                # 验证成功，开始生成托盘
                self.state = 'GENERATE_PALLETS'
            else:
                # 验证失败，重试
                self.camera_retry_count += 1
                if self.camera_retry_count >= self.max_camera_retries:
                    self.get_logger().error(
                        f'Failed to set camera position after {self.max_camera_retries} retries. '
                        f'Trying new random position...'
                    )
                    self.state = 'SET_CAMERA'
                else:
                    self.get_logger().info(f'Retrying camera position ({self.camera_retry_count}/{self.max_camera_retries})...')
                    # 重新发送设置命令
                    self.set_entity_pose(
                        self.camera_name,
                        self.target_camera_x, 
                        self.target_camera_y, 
                        self.target_camera_z,
                        0.0, 0.0, self.target_camera_yaw
                    )
            
        elif self.state == 'GENERATE_PALLETS':
            # 生成托盘
            if self.pallet_count >= self.pallets_per_camera:
                self.get_logger().info(
                    f'Completed {self.pallet_count} pallets. Moving to new camera position...\n'
                )
                self.state = 'SET_CAMERA'
                return
            
            pallet_x, pallet_y, pallet_z, pallet_yaw = self.generate_pallet_in_fov(
                self.target_camera_x, 
                self.target_camera_y, 
                self.target_camera_z, 
                self.target_camera_yaw
            )
            
            self.set_entity_pose(
                self.pallet_name,
                pallet_x, pallet_y, pallet_z,
                0.0, 0.0, pallet_yaw
            )
            
            distance = math.sqrt(
                (self.target_camera_x - pallet_x)**2 + 
                (self.target_camera_y - pallet_y)**2 + 
                (self.target_camera_z - pallet_z)**2
            )
            
            self.pallet_count += 1
            
            self.get_logger().info(
                f'[{self.pallet_count}/{self.pallets_per_camera}] '
                f'Pallet: ({pallet_x:.2f}, {pallet_y:.2f}), '
                f'yaw={math.degrees(pallet_yaw):.1f}°, '
                f'Distance: {distance:.2f}m'
            )


    def generate_pallet_in_fov(self, camera_x, camera_y, camera_z, camera_yaw):
        """根据相机位置和朝向，在视野内生成托盘位置，并确保托盘朝向角度限制"""
        pallet_z = 0.0
        distance = random.uniform(self.pallet_distance_min, self.pallet_distance_max)
        
        height_diff = camera_z - pallet_z
        horizontal_distance = math.sqrt(max(0, distance**2 - height_diff**2))
        
        if horizontal_distance < 0.5:
            horizontal_distance = 0.5
        
        fov_rad = math.radians(self.camera_fov * 0.8)
        angle_offset = random.uniform(-fov_rad/2, fov_rad/2)
        camera_forward_yaw = camera_yaw #- math.pi
        pallet_angle = camera_forward_yaw + angle_offset
        
        pallet_x = camera_x + horizontal_distance * math.cos(pallet_angle)
        pallet_y = camera_y + horizontal_distance * math.sin(pallet_angle)
        
        pallet_x = max(self.warehouse_x_range[0] + 0.5, 
                      min(pallet_x, self.warehouse_x_range[1] - 0.5))
        pallet_y = max(self.warehouse_y_range[0] + 0.5, 
                      min(pallet_y, self.warehouse_y_range[1] - 0.5))
        
        # 计算从托盘指向相机的方向角
        angle_to_camera = math.atan2(camera_y - pallet_y, camera_x - pallet_x)
        
        # 在允许的角度范围内随机生成托盘yaw角
        # 托盘yaw相对于朝向相机的角度，偏移范围为 [-60度, +60度]
        yaw_offset = random.uniform(-self.pallet_max_angle, self.pallet_max_angle)
        pallet_yaw = angle_to_camera + yaw_offset
        
        # 归一化到 [-π, π]
        pallet_yaw = math.atan2(math.sin(pallet_yaw), math.cos(pallet_yaw))
        
        return pallet_x, pallet_y, pallet_z, pallet_yaw

    def set_entity_pose(self, entity_name, x, y, z, roll, pitch, yaw):
        """设置实体的位姿"""
        qx, qy, qz, qw = self.euler_to_quaternion(roll, pitch, yaw)
        
        state = EntityState()
        state.name = entity_name
        state.pose = Pose()
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = z
        state.pose.orientation.x = qx
        state.pose.orientation.y = qy
        state.pose.orientation.z = qz
        state.pose.orientation.w = qw
        
        req = SetEntityState.Request()
        req.state = state
        
        try:
            future = self.set_state_cli.call_async(req)
            return True
        except Exception as e:
            self.get_logger().error(f'Failed to call set_entity_state service: {e}')
            return False

    @staticmethod
    def quaternion_to_yaw(x, y, z, w):
        """将四元数转换为yaw角"""
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    @staticmethod
    def angle_diff(a, b):
        """计算两个角度的最小差值"""
        diff = a - b
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff

    @staticmethod
    def euler_to_quaternion(roll, pitch, yaw):
        """将欧拉角转换为四元数"""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return qx, qy, qz, qw


def main(args=None):
    rclpy.init(args=args)
    node = RandomPalletPose()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

