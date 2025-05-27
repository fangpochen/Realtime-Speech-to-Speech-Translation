#!/usr/bin/env python3
"""简单的麦克风测试程序"""
import speech_recognition as sr
import sounddevice as sd
import numpy as np
import time

def test_microphone():
    """测试麦克风是否正常工作"""
    print("=== 麦克风测试程序 ===")
    
    # 显示音频设备
    print("\n可用音频设备:")
    print(sd.query_devices())
    
    # 获取默认设备
    input_device, output_device = sd.default.device
    print(f"\n默认输入设备: {input_device}")
    print(f"默认输出设备: {output_device}")
    
    # 初始化语音识别
    r = sr.Recognizer()
    r.energy_threshold = 300  # 降低阈值
    r.dynamic_energy_threshold = False
    r.pause_threshold = 0.8
    
    # 使用麦克风
    mic = sr.Microphone(device_index=input_device, sample_rate=16000)
    
    print("\n正在调整环境噪音...")
    with mic as source:
        r.adjust_for_ambient_noise(source)
    
    print(f"能量阈值设置为: {r.energy_threshold}")
    print("\n开始监听麦克风... (说话测试)")
    print("按 Ctrl+C 退出")
    
    def callback(recognizer, audio):
        try:
            # 获取音频数据
            audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
            volume = np.sqrt(np.mean(audio_data**2))
            print(f"检测到音频! 音量: {volume:.2f}, 数据长度: {len(audio_data)}")
            
            # 发送到服务器测试
            import socket
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(('localhost', 4444))
                sock.send(audio.get_raw_data())
                print("音频数据已发送到服务器")
                sock.close()
            except Exception as e:
                print(f"发送到服务器失败: {e}")
                
        except Exception as e:
            print(f"处理音频时出错: {e}")
    
    # 开始后台监听
    stop_listening = r.listen_in_background(mic, callback, phrase_time_limit=None)
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n停止监听...")
        stop_listening(wait_for_stop=False)

if __name__ == "__main__":
    test_microphone() 