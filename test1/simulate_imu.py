import socket
import json
import time
import random
import math  # 核心修复：全局导入math模块
from typing import Dict, Any

# 模拟ICM42688传感器参数（匹配硬件配置）
GRAVITY = 9.807  # 重力加速度 (m/s²)
GYRO_NOISE = 0.05  # 陀螺仪噪声 (dps)
ACCEL_NOISE = 0.02  # 加速度计噪声 (m/s²)
UPDATE_FREQ = 100  # 模拟数据更新频率 (Hz)
UDP_HOST = "127.0.0.1"  # 后端接收地址
UDP_PORT = 12345  # 后端UDP端口

class IMUSimulator:
    def __init__(self):
        # 初始姿态（静止状态，轻微偏移模拟真实硬件）
        self.accel_x = 0.0 + random.uniform(-ACCEL_NOISE, ACCEL_NOISE)
        self.accel_y = 0.0 + random.uniform(-ACCEL_NOISE, ACCEL_NOISE)
        self.accel_z = GRAVITY + random.uniform(-ACCEL_NOISE, ACCEL_NOISE)
        
        self.gyro_x = 0.0 + random.uniform(-GYRO_NOISE, GYRO_NOISE)
        self.gyro_y = 0.0 + random.uniform(-GYRO_NOISE, GYRO_NOISE)
        self.gyro_z = 0.0 + random.uniform(-GYRO_NOISE, GYRO_NOISE)
        
        self.roll = 0.0  # 横滚角 (°)
        self.pitch = 0.0  # 俯仰角 (°)
        self.yaw = 0.0  # 偏航角 (°)
        
        # 动态变化参数（模拟轻微运动）
        self.angle_step = 0.1  # 角度变化步长
        self.dir_change_interval = 5  # 方向变化间隔 (s)
        self.last_dir_change = time.time()
        self.roll_dir = 1
        self.pitch_dir = 1
        self.yaw_dir = 1

    def _add_noise(self, value: float, noise: float) -> float:
        """为传感器数据添加高斯噪声"""
        return value + random.gauss(0, noise)

    def update_imu_data(self):
        """模拟IMU数据动态变化（每帧更新）"""
        current_time = time.time()
        
        # 每隔指定时间随机改变角度变化方向
        if current_time - self.last_dir_change > self.dir_change_interval:
            self.roll_dir = random.choice([-1, 0, 1])
            self.pitch_dir = random.choice([-1, 0, 1])
            self.yaw_dir = random.choice([-1, 0, 1])
            self.last_dir_change = current_time
        
        # 更新欧拉角（模拟缓慢姿态变化）
        self.roll += self.roll_dir * self.angle_step
        self.pitch += self.pitch_dir * self.angle_step
        self.yaw += self.yaw_dir * self.angle_step
        
        # 限制角度范围（-180° ~ 180°）
        self.roll = (self.roll + 180) % 360 - 180
        self.pitch = (self.pitch + 180) % 360 - 180
        self.yaw = (self.yaw + 180) % 360 - 180
        
        # 根据欧拉角更新加速度计数据（模拟重力分量）
        roll_rad = math.radians(self.roll)
        pitch_rad = math.radians(self.pitch)
        
        self.accel_x = GRAVITY * math.sin(pitch_rad)
        self.accel_y = GRAVITY * math.sin(roll_rad) * math.cos(pitch_rad)
        self.accel_z = GRAVITY * math.cos(roll_rad) * math.cos(pitch_rad)
        
        # 根据角度变化率更新陀螺仪数据
        self.gyro_x = self.roll_dir * self.angle_step * UPDATE_FREQ
        self.gyro_y = self.pitch_dir * self.angle_step * UPDATE_FREQ
        self.gyro_z = self.yaw_dir * self.angle_step * UPDATE_FREQ

    def get_data(self) -> Dict[str, Any]:
        """生成符合后端格式要求的IMU数据"""
        self.update_imu_data()
        
        # 生成时间戳（毫秒级，匹配后端要求）
        ts = time.time() * 1000
        
        return {
            "ts": ts,
            "accel": {
                "x": round(self._add_noise(self.accel_x, ACCEL_NOISE), 2),
                "y": round(self._add_noise(self.accel_y, ACCEL_NOISE), 2),
                "z": round(self._add_noise(self.accel_z, ACCEL_NOISE), 2)
            },
            "gyro": {
                "x": round(self._add_noise(self.gyro_x, GYRO_NOISE), 2),
                "y": round(self._add_noise(self.gyro_y, GYRO_NOISE), 2),
                "z": round(self._add_noise(self.gyro_z, GYRO_NOISE), 2)
            },
            "angles": {
                "roll": round(self.roll, 2),
                "pitch": round(self.pitch, 2),
                "yaw": round(self.yaw, 2)
            },
            "temp": round(25.0 + random.uniform(-1.0, 1.0), 2)  # 模拟温度数据
        }

    def send_via_udp(self, host: str = UDP_HOST, port: int = UDP_PORT) -> None:
        """通过UDP发送IMU数据到后端"""
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            data = self.get_data()
            json_data = json.dumps(data)
            udp_socket.sendto(json_data.encode("utf-8"), (host, port))
            # 打印发送日志（可选，调试用）
            print(f"[IMU Simulator] 发送数据: {json_data[:100]}...", flush=True)
        except Exception as e:
            print(f"[IMU Simulator] UDP发送失败: {e}", flush=True)
        finally:
            udp_socket.close()

def main():
    """主函数：持续生成并发送IMU模拟数据"""
    simulator = IMUSimulator()
    interval = 1.0 / UPDATE_FREQ  # 发送间隔（秒）
    
    print(f"[IMU Simulator] 启动中... | 目标地址: {UDP_HOST}:{UDP_PORT} | 频率: {UPDATE_FREQ}Hz")
    print(f"[IMU Simulator] 按 Ctrl+C 停止")
    
    try:
        while True:
            start_time = time.time()
            simulator.send_via_udp()
            # 控制发送频率，补偿处理耗时
            elapsed = time.time() - start_time
            if elapsed < interval:
                time.sleep(interval - elapsed)
    except KeyboardInterrupt:
        print("\n[IMU Simulator] 已停止")
    except Exception as e:
        print(f"\n[IMU Simulator] 异常停止: {e}")

if __name__ == "__main__":
    main()