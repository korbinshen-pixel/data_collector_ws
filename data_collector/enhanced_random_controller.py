#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState, DeleteEntity, SpawnEntity
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Pose
import random
import math


class EnhancedRandomController(Node):
    def __init__(self):
        super().__init__('enhanced_random_controller')

        # 创建服务客户端
        self.set_state_cli = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.delete_cli = self.create_client(DeleteEntity, '/gazebo/delete_entity')
        self.spawn_cli = self.create_client(SpawnEntity, '/gazebo/spawn_entity')
        
        # 等待服务可用
        while not self.set_state_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /gazebo/set_entity_state service...')

        # 模型名称
        self.target_name = 'pallet'
        
        # 定义常见托盘颜色预设
        self.color_presets = [
            (0.6, 0.4, 0.2),   # 棕色
            (0.5, 0.5, 0.5),   # 灰色
            (0.2, 0.3, 0.6),   # 蓝色
            (0.3, 0.5, 0.3),   # 绿色
            (0.7, 0.6, 0.4),   # 浅棕色
            (0.3, 0.3, 0.3),   # 深灰色
            (0.8, 0.5, 0.3),   # 橙棕色
            (0.4, 0.4, 0.5),   # 蓝灰色
        ]
        
        # 托盘SDF模板
        self.pallet_sdf_template = '''<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="pallet">
    <static>false</static>
    <link name="pallet_link">
      <pose>0 0 0.075 0 0 0</pose>
      <inertial>
        <mass>10.0</mass>
        <inertia>
          <ixx>1</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>1</iyy><iyz>0</iyz><izz>1</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <box><size>1.2 1.0 0.15</size></box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box><size>1.2 1.0 0.15</size></box>
        </geometry>
        <material>
          <ambient>{r} {g} {b} 1</ambient>
          <diffuse>{r} {g} {b} 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>'''
        
        self.needs_respawn = False
        self.pending_pose = None
        
        # 每隔 0.5 秒调用一次 timer_callback
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.get_logger().info('Enhanced random controller node started.')

    def timer_callback(self):
        """定时器回调，随机生成托盘的位置、姿态和颜色"""
        
        # 生成随机位姿
        x = random.uniform(-2.5, 2.5)
        y = random.uniform(-2.5, 2.5)
        z = 0.0
        yaw = random.uniform(-math.pi, math.pi)
        qx, qy, qz, qw = self.yaw_to_quat(yaw)

        # 创建位姿
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw

        # 每10次改变一次颜色（删除并重新spawn）
        if random.random() < 0.1:  # 10%概率改变颜色
            self.pending_pose = pose
            self.respawn_with_new_color()
        else:
            # 只改变位姿
            self.update_pose(pose)

    def update_pose(self, pose):
        """更新托盘位姿"""
        state = EntityState()
        state.name = self.target_name
        state.pose = pose

        req = SetEntityState.Request()
        req.state = state

        future = self.set_state_cli.call_async(req)
        future.add_done_callback(self.pose_update_done)

    def pose_update_done(self, future):
        """位姿更新完成回调"""
        try:
            resp = future.result()
            if resp.success:
                self.get_logger().debug('Updated pallet pose')
        except Exception as e:
            self.get_logger().error(f'Pose update failed: {e}')

    def respawn_with_new_color(self):
        """删除并重新spawn新颜色的托盘"""
        # 先删除
        req = DeleteEntity.Request()
        req.name = self.target_name
        future = self.delete_cli.call_async(req)
        future.add_done_callback(self.delete_done)

    def delete_done(self, future):
        """删除完成，spawn新托盘"""
        try:
            future.result()
            self.get_logger().info('Deleted old pallet')
        except Exception as e:
            self.get_logger().warn(f'Delete may have failed: {e}')
        
        # Spawn新颜色的托盘
        self.spawn_new_pallet()

    def spawn_new_pallet(self):
        """Spawn新颜色的托盘"""
        # 生成随机颜色
        r, g, b = self.get_random_color()
        sdf = self.pallet_sdf_template.format(r=r, g=g, b=b)

        req = SpawnEntity.Request()
        req.name = self.target_name
        req.xml = sdf
        req.initial_pose = self.pending_pose if self.pending_pose else Pose()

        future = self.spawn_cli.call_async(req)
        future.add_done_callback(self.spawn_done)

    def spawn_done(self, future):
        """Spawn完成回调"""
        try:
            resp = future.result()
            if resp.success:
                self.get_logger().info('Spawned pallet with new color')
        except Exception as e:
            self.get_logger().error(f'Spawn failed: {e}')

    def get_random_color(self):
        """生成随机托盘颜色"""
        r, g, b = random.choice(self.color_presets)
        r = max(0.1, min(0.9, r + random.uniform(-0.1, 0.1)))
        g = max(0.1, min(0.9, g + random.uniform(-0.1, 0.1)))
        b = max(0.1, min(0.9, b + random.uniform(-0.1, 0.1)))
        return r, g, b

    @staticmethod
    def yaw_to_quat(yaw):
        """将绕 z 轴的旋转角（yaw）转换为四元数"""
        half_yaw = yaw / 2.0
        qz = math.sin(half_yaw)
        qw = math.cos(half_yaw)
        return 0.0, 0.0, qz, qw


def main(args=None):
    rclpy.init(args=args)
    node = EnhancedRandomController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

