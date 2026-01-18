# local_devices.py
# -*- coding: utf-8 -*-
"""
本地设备模块：提供IP摄像头、本地麦克风和本地扬声器支持
用于调试模式，当ESP32设备未连接时自动切换到本地设备
"""

import os
import sys
import time
import asyncio
import threading
import queue
from typing import Optional, Callable
import cv2
import numpy as np

# 确保在读取环境变量前加载.env文件
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# 音频支持
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("[LOCAL_DEVICES] 警告: pyaudio未安装，本地音频功能不可用")

# ============ 配置 ============
IP_CAMERA_URL = os.getenv("IP_CAMERA_URL", "http://192.168.101.31:8081/video")
IP_CAMERA_USER = os.getenv("IP_CAMERA_USER", "admin")
IP_CAMERA_PASS = os.getenv("IP_CAMERA_PASS", "admin")
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
USE_LOCAL_AUDIO = os.getenv("USE_LOCAL_AUDIO", "false").lower() == "true"

# 打印配置信息（调试用）
print(f"[LOCAL_DEVICES] DEBUG_MODE={DEBUG_MODE}, USE_LOCAL_AUDIO={USE_LOCAL_AUDIO}")
print(f"[LOCAL_DEVICES] IP_CAMERA_URL={IP_CAMERA_URL}")

# 音频参数（与ESP32保持一致）
SAMPLE_RATE = 16000
CHUNK_MS = 20
BYTES_PER_CHUNK = SAMPLE_RATE * CHUNK_MS // 1000 * 2  # 640 bytes (16bit mono)
AUDIO_FORMAT = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None
CHANNELS = 1


class IPCameraCapture:
    """IP摄像头采集器"""
    
    def __init__(self, url: str = None, username: str = None, password: str = None):
        self.url = url or IP_CAMERA_URL
        self.username = username or IP_CAMERA_USER
        self.password = password or IP_CAMERA_PASS
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_callback: Optional[Callable[[bytes], None]] = None
        self._last_frame: Optional[bytes] = None
        self._lock = threading.Lock()
        self._frame_count = 0
        self._last_log_time = 0
        
    def _build_url(self) -> str:
        """构建带认证的URL"""
        if self.username and self.password:
            # 将认证信息嵌入URL
            if "://" in self.url:
                protocol, rest = self.url.split("://", 1)
                return f"{protocol}://{self.username}:{self.password}@{rest}"
        return self.url
    
    def start(self, frame_callback: Callable[[bytes], None] = None):
        """启动IP摄像头采集"""
        if self._running:
            print("[IP_CAMERA] 已在运行中")
            return True
            
        self._frame_callback = frame_callback
        
        # 尝试连接摄像头
        url = self._build_url()
        print(f"[IP_CAMERA] 正在连接: {self.url}")
        
        try:
            self._cap = cv2.VideoCapture(url)
            if not self._cap.isOpened():
                print(f"[IP_CAMERA] 无法打开摄像头: {self.url}")
                return False
                
            # 设置缓冲区大小，减少延迟
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            print("[IP_CAMERA] 摄像头采集已启动")
            return True
            
        except Exception as e:
            print(f"[IP_CAMERA] 启动失败: {e}")
            return False
    
    def _capture_loop(self):
        """采集循环"""
        consecutive_failures = 0
        max_failures = 30  # 连续失败30次后重连
        
        while self._running:
            try:
                if self._cap is None or not self._cap.isOpened():
                    print("[IP_CAMERA] 尝试重新连接...")
                    time.sleep(1)
                    url = self._build_url()
                    self._cap = cv2.VideoCapture(url)
                    if not self._cap.isOpened():
                        consecutive_failures += 1
                        continue
                    consecutive_failures = 0
                
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        print("[IP_CAMERA] 连续读取失败，尝试重连")
                        if self._cap:
                            self._cap.release()
                        self._cap = None
                        consecutive_failures = 0
                    time.sleep(0.01)
                    continue
                
                consecutive_failures = 0
                self._frame_count += 1
                
                # 编码为JPEG
                ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ok:
                    jpeg_data = enc.tobytes()
                    
                    with self._lock:
                        self._last_frame = jpeg_data
                    
                    # 回调
                    if self._frame_callback:
                        try:
                            self._frame_callback(jpeg_data)
                        except Exception as e:
                            if self._frame_count % 100 == 0:
                                print(f"[IP_CAMERA] 回调错误: {e}")
                
                # 日志
                current_time = time.time()
                if current_time - self._last_log_time > 10:
                    print(f"[IP_CAMERA] 运行中，已采集 {self._frame_count} 帧")
                    self._last_log_time = current_time
                    
            except Exception as e:
                print(f"[IP_CAMERA] 采集错误: {e}")
                time.sleep(0.1)
    
    def get_last_frame(self) -> Optional[bytes]:
        """获取最新帧"""
        with self._lock:
            return self._last_frame
    
    def stop(self):
        """停止采集"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        print("[IP_CAMERA] 摄像头采集已停止")
    
    def is_running(self) -> bool:
        return self._running


class LocalMicrophone:
    """本地麦克风录音器"""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE, chunk_ms: int = CHUNK_MS):
        if not PYAUDIO_AVAILABLE:
            raise RuntimeError("pyaudio未安装，无法使用本地麦克风")
            
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.chunk_size = sample_rate * chunk_ms // 1000  # samples per chunk
        
        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._audio_callback: Optional[Callable[[bytes], None]] = None
        self._audio_queue: queue.Queue = queue.Queue(maxsize=50)
        
    def start(self, audio_callback: Callable[[bytes], None] = None):
        """启动麦克风录音"""
        if self._running:
            print("[LOCAL_MIC] 已在运行中")
            return True
            
        self._audio_callback = audio_callback
        
        try:
            self._pa = pyaudio.PyAudio()
            
            # 查找默认输入设备
            default_input = self._pa.get_default_input_device_info()
            print(f"[LOCAL_MIC] 使用设备: {default_input['name']}")
            
            self._stream = self._pa.open(
                format=AUDIO_FORMAT,
                channels=CHANNELS,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_input_callback
            )
            
            self._running = True
            self._stream.start_stream()
            print(f"[LOCAL_MIC] 麦克风录音已启动 ({self.sample_rate}Hz, {self.chunk_ms}ms chunks)")
            return True
            
        except Exception as e:
            print(f"[LOCAL_MIC] 启动失败: {e}")
            self._cleanup()
            return False
    
    def _audio_input_callback(self, in_data, frame_count, time_info, status):
        """pyaudio回调"""
        if self._running and in_data:
            if self._audio_callback:
                try:
                    self._audio_callback(in_data)
                except Exception:
                    pass
            
            # 也放入队列供同步读取
            try:
                self._audio_queue.put_nowait(in_data)
            except queue.Full:
                pass
                
        return (None, pyaudio.paContinue)
    
    def read_chunk(self, timeout: float = 0.1) -> Optional[bytes]:
        """同步读取一个音频块"""
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop(self):
        """停止录音"""
        self._running = False
        self._cleanup()
        print("[LOCAL_MIC] 麦克风录音已停止")
    
    def _cleanup(self):
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._pa:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None
    
    def is_running(self) -> bool:
        return self._running


class LocalSpeaker:
    """本地扬声器播放器"""
    
    def __init__(self, sample_rate: int = 8000):
        if not PYAUDIO_AVAILABLE:
            raise RuntimeError("pyaudio未安装，无法使用本地扬声器")
            
        self.sample_rate = sample_rate
        
        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream = None
        self._running = False
        self._audio_queue: queue.Queue = queue.Queue(maxsize=100)
        self._thread: Optional[threading.Thread] = None
        
    def start(self):
        """启动扬声器播放"""
        if self._running:
            print("[LOCAL_SPK] 已在运行中")
            return True
            
        try:
            self._pa = pyaudio.PyAudio()
            
            # 查找默认输出设备
            default_output = self._pa.get_default_output_device_info()
            print(f"[LOCAL_SPK] 使用设备: {default_output['name']}")
            
            self._stream = self._pa.open(
                format=AUDIO_FORMAT,
                channels=CHANNELS,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=1024
            )
            
            self._running = True
            self._thread = threading.Thread(target=self._playback_loop, daemon=True)
            self._thread.start()
            print(f"[LOCAL_SPK] 扬声器播放已启动 ({self.sample_rate}Hz)")
            return True
            
        except Exception as e:
            print(f"[LOCAL_SPK] 启动失败: {e}")
            self._cleanup()
            return False
    
    def _playback_loop(self):
        """播放循环"""
        while self._running:
            try:
                # 从队列获取音频数据
                audio_data = self._audio_queue.get(timeout=0.1)
                if audio_data and self._stream:
                    self._stream.write(audio_data)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[LOCAL_SPK] 播放错误: {e}")
    
    def play(self, audio_data: bytes):
        """播放音频数据（异步）"""
        if not self._running:
            return
            
        try:
            # 如果队列接近满，清空旧数据保持实时性
            if self._audio_queue.qsize() > 80:
                while not self._audio_queue.empty():
                    try:
                        self._audio_queue.get_nowait()
                    except queue.Empty:
                        break
            
            self._audio_queue.put_nowait(audio_data)
        except queue.Full:
            pass
    
    def play_sync(self, audio_data: bytes):
        """同步播放音频数据"""
        if self._stream:
            try:
                self._stream.write(audio_data)
            except Exception as e:
                print(f"[LOCAL_SPK] 同步播放错误: {e}")
    
    def stop(self):
        """停止播放"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._cleanup()
        print("[LOCAL_SPK] 扬声器播放已停止")
    
    def _cleanup(self):
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._pa:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None
    
    def is_running(self) -> bool:
        return self._running


# ============ 全局实例和管理 ============
_ip_camera: Optional[IPCameraCapture] = None
_local_mic: Optional[LocalMicrophone] = None
_local_speaker: Optional[LocalSpeaker] = None


def get_ip_camera() -> Optional[IPCameraCapture]:
    """获取IP摄像头实例"""
    global _ip_camera
    if _ip_camera is None:
        _ip_camera = IPCameraCapture()
    return _ip_camera


def get_local_microphone() -> Optional[LocalMicrophone]:
    """获取本地麦克风实例"""
    global _local_mic
    if _local_mic is None and PYAUDIO_AVAILABLE:
        _local_mic = LocalMicrophone()
    return _local_mic


def get_local_speaker() -> Optional[LocalSpeaker]:
    """获取本地扬声器实例"""
    global _local_speaker
    if _local_speaker is None and PYAUDIO_AVAILABLE:
        _local_speaker = LocalSpeaker()
    return _local_speaker


def is_debug_mode() -> bool:
    """检查是否启用调试模式"""
    return DEBUG_MODE


def is_local_audio_enabled() -> bool:
    """检查是否启用本地音频"""
    return USE_LOCAL_AUDIO and PYAUDIO_AVAILABLE


def cleanup_all():
    """清理所有本地设备"""
    global _ip_camera, _local_mic, _local_speaker
    
    if _ip_camera:
        _ip_camera.stop()
        _ip_camera = None
    if _local_mic:
        _local_mic.stop()
        _local_mic = None
    if _local_speaker:
        _local_speaker.stop()
        _local_speaker = None
    
    print("[LOCAL_DEVICES] 所有本地设备已清理")


# ============ 测试入口 ============
if __name__ == "__main__":
    print("=" * 50)
    print("本地设备测试")
    print("=" * 50)
    
    # 测试IP摄像头
    print("\n[测试1] IP摄像头...")
    camera = IPCameraCapture()
    
    frame_count = 0
    def on_frame(jpeg_data):
        global frame_count
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  收到帧 {frame_count}, 大小: {len(jpeg_data)} bytes")
    
    if camera.start(on_frame):
        time.sleep(5)
        camera.stop()
        print(f"  共收到 {frame_count} 帧")
    else:
        print("  IP摄像头启动失败")
    
    # 测试本地麦克风
    if PYAUDIO_AVAILABLE:
        print("\n[测试2] 本地麦克风...")
        mic = LocalMicrophone()
        
        audio_chunks = 0
        def on_audio(audio_data):
            global audio_chunks
            audio_chunks += 1
            if audio_chunks % 50 == 0:
                print(f"  收到音频块 {audio_chunks}, 大小: {len(audio_data)} bytes")
        
        if mic.start(on_audio):
            time.sleep(3)
            mic.stop()
            print(f"  共收到 {audio_chunks} 个音频块")
        else:
            print("  麦克风启动失败")
        
        # 测试本地扬声器
        print("\n[测试3] 本地扬声器...")
        speaker = LocalSpeaker()
        if speaker.start():
            # 生成测试音（440Hz正弦波）
            import math
            duration = 1  # 1秒
            samples = []
            for i in range(8000 * duration):
                sample = int(32767 * 0.5 * math.sin(2 * math.pi * 440 * i / 8000))
                samples.append(sample)
            
            import struct
            audio_data = struct.pack(f"<{len(samples)}h", *samples)
            speaker.play_sync(audio_data)
            time.sleep(0.5)
            speaker.stop()
            print("  扬声器测试完成（应该听到1秒440Hz音调）")
        else:
            print("  扬声器启动失败")
    else:
        print("\n[跳过] 本地音频测试（pyaudio未安装）")
    
    print("\n测试完成!")
