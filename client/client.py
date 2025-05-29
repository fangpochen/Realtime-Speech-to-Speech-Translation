"""Client for live speech to speech translation"""
import logging
import socket
import time
import threading
from datetime import datetime, timezone
import speech_recognition as sr
import numpy as np
import sounddevice as sd
import struct # <-- 添加导入
import wave  # <-- 添加导入
import io    # <-- 添加导入
import librosa # <--- 添加librosa导入
from utils.print_audio import print_sound, get_volume_norm, convert_and_normalize
import os # 导入os模块

HEADER_LENGTH = 8 # 定义头部长度为常量

class AudioSocketClient:
    """ Client for recording audio, streaming it to the server via sockets, receiving
    the data and then piping it to an output audio device """
    CHANNELS = 1
    RECORDER_RATE = 16000 # 采样率给ASR模型
    PLAYBACK_RATE = 32000 # 采样率用于播放接收到的TTS音频 (基于假设)
    CHUNK = 4096
    # Used for Speech Recognition library - set this higher for non-English languages
    PHRASE_TIME_LIMIT = 3  # 增加到3秒，给更多时间说话
    # How long you need to stop speaking to be considered an entire phrase
    PAUSE_THRESHOLD = 1.0  # 增加停顿检测时间，更容易检测到停顿
    # Volume for the microphone (降低阈值以提高敏感度)
    RECORDER_ENERGY_THRESHOLD = 800
    def __init__(self) -> None:
        # Prompt the user to select their devices
        self.input_device_index, self.output_device_index = sd.default.device
        print(sd.query_devices())
        print(f"Using input index of: {self.input_device_index}\noutput index of: {self.output_device_index}.")
        if input(" Is this correct?\n y/[n]: ") != "y":
            self.input_device_index = int(input("Type the index of the physical microphone: "))
            self.output_device_index = int(input("Type the index of the output microphone: "))

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = self.RECORDER_ENERGY_THRESHOLD
        """Definitely do this, dynamic energy compensation lowers the energy threshold dramatically 
            to a point where the SpeechRecognizer never stops recording."""
        self.recorder.dynamic_energy_threshold = False
        self.recorder.pause_threshold = self.PAUSE_THRESHOLD
        # 使用RECORDER_RATE进行录音
        self.source = sr.Microphone(device_index=self.input_device_index, sample_rate=self.RECORDER_RATE)
        self.transcription = [""]
        ### Debugging variables
        self.time_last_sent = None
        self.time_first_received = None
        self.time_last_received = None
        self.volume_input = 0
        self.volume_output = 0
        # How much time since the last received packet to refresh the flush
        self.time_flush_received = 2
        self.time_phrase_sent = None # 用于记录短语发送时间以计算延迟
        threading.Thread(target=self.__debug_worker__, daemon=True).start()
    def __del__(self):
        # Destroy Audio resources
        print('Shutting down')

    def record_callback(self, _, audio: sr.AudioData):
        """ Callback function for microphone input, 
            fires when there is new data from the microphone """
        data = audio.get_raw_data()
        self.time_last_sent = time.time()
        logging.debug("send audio data %f", self.time_last_sent)
        self.socket.send(data)
        self.time_phrase_sent = time.time() # 记录短语发送时间
        # convert to np array for volume
        self.volume_input = get_volume_norm(
            convert_and_normalize(np.frombuffer(data, dtype=np.int16))
        )

    def _recv_all_data(self, sock, n_bytes):
        """辅助函数：确保从socket接收指定数量的字节，或者在连接关闭时返回None。"""
        buffer = b''
        while len(buffer) < n_bytes:
            try:
                packet = sock.recv(min(n_bytes - len(buffer), self.CHUNK))
            except ConnectionResetError:
                print("❌ 在 _recv_all_data 中连接被重置")
                return None
            except socket.error as e:
                print(f"❌ 在 _recv_all_data 中发生socket错误: {e}")
                return None
            
            if not packet: # 套接字已关闭
                print("ℹ️ 在 _recv_all_data 中检测到socket已关闭 (recv返回空)")
                return None 
            buffer += packet
        return buffer

    def start(self, ip, port):
        """ Starts the client service """
        # Connect to server
        print(f"Attempting to connect to IP {ip}, port {port}")
        self.socket.connect((ip, port))
        print(f"Successfully connected to IP {ip}, port {port}.")

        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)
        self.recorder.listen_in_background(self.source,
                                           self.record_callback,
                                           phrase_time_limit=self.PHRASE_TIME_LIMIT)
        print('''Listening now...\nNote: The input microphone records
              in very large packets, so the volume meter won't move as much.''')
        self.volume_print_worker = threading.Thread(target=self.__volume_print_worker__,
                                                    daemon=True)
        self.volume_print_worker.start()

        # OutputStream应该在最外层，因为它管理音频输出设备
        # 使用PLAYBACK_RATE进行播放
        with sd.OutputStream(samplerate=self.PLAYBACK_RATE, 
                    channels=self.CHANNELS, 
                    dtype=np.float32, 
                    device=self.output_device_index,
                    ) as audio_output: 
            try:
                while True: 
                    print("🎧 等待接收服务端音频头部...")
                    
                    # 1. 接收数据长度头部
                    header_bytes = self._recv_all_data(self.socket, HEADER_LENGTH)
                    if header_bytes is None:
                        print("🚫 接收头部失败或连接已关闭。客户端将退出。")
                        break # 跳出主循环
                    
                    try:
                        audio_data_length = struct.unpack("!Q", header_bytes)[0]
                        print(f"📨 收到头部，预期音频数据长度: {audio_data_length} bytes")
                    except struct.error as e_unpack:
                        print(f"❌ 解包头部失败: {e_unpack}。接收到的头部: {header_bytes!r}")
                        break # 跳出主循环

                    if audio_data_length == 0:
                        print("ℹ️  收到长度为0的音频数据，视为空消息，继续等待。")
                        continue # 继续外层循环，等待下一个头部

                    # 2. 接收实际的音频数据
                    print(f"⬇️ 开始接收 {audio_data_length} bytes 的音频数据...")
                    full_received_data = self._recv_all_data(self.socket, audio_data_length)
                    
                    if full_received_data is None:
                        print(f"🚫 接收 {audio_data_length} bytes 的音频数据失败或连接中途关闭。客户端将退出。")
                        break # 跳出主循环
                    
                    if len(full_received_data) < audio_data_length:
                        print(f"⚠️ 接收到的音频数据不完整。预期 {audio_data_length}, 收到 {len(full_received_data)}. 将尝试处理已接收部分。")
                        # 这种情况理论上不应由_recv_all_data返回，除非_recv_all_data逻辑有误或中途发生非致命错误
                        # 但为保险起见，保留一个检查和日志
                    
                    # time_audio_received = time.time() # 记录音频接收时间 # 旧的逻辑，确保它不干扰新的计时
                    # if self.time_phrase_sent: # 旧的逻辑
                    #     latency = time_audio_received - self.time_phrase_sent
                    #     print(f"⏱️ 音频处理延迟: {latency:.3f} 秒 (从发送到接收)")
                    #     self.time_phrase_sent = None # 旧的重置位置
                    
                    timestamp = int(time.time())
                    print(f"🟢 完整音频数据接收完毕 (批次 {timestamp})，总大小: {len(full_received_data)} bytes")
                    output_filename = f"client_received_audio_{timestamp}.wav"
                    try:
                        with open(output_filename, 'wb') as f_out:
                            f_out.write(full_received_data)
                        print(f"💾 客户端接收的音频已保存到: {output_filename}")
                    except Exception as e_save:
                        print(f"❌ 保存客户端接收的音频失败: {e_save}")
                    
                    # 开始播放接收到的音频
                    print(f"▶️ 尝试播放接收到的音频: {output_filename}")
                    try:
                        with io.BytesIO(full_received_data) as wav_bytes_io:
                            with wave.open(wav_bytes_io, 'rb') as wf:
                                wav_framerate = wf.getframerate()
                                wav_channels = wf.getnchannels()
                                wav_sampwidth = wf.getsampwidth()
                                num_frames = wf.getnframes()
                                pcm_data_bytes = wf.readframes(num_frames)

                                print(f"   [WAV Info] 文件采样率: {wav_framerate}, 声道数: {wav_channels}, 位深: {wav_sampwidth*8}-bit, 帧数: {num_frames}")

                                if wav_channels != self.CHANNELS:
                                    print(f"❌ 错误: WAV文件声道数 ({wav_channels}) 与播放器预设声道数 ({self.CHANNELS}) 不匹配! 无法正确播放.")
                                elif wav_sampwidth != 2: # 2 bytes = 16-bit PCM
                                    print(f"❌ 错误: WAV文件样本宽度 ({wav_sampwidth} bytes) 不是预期的2 bytes (16-bit PCM)! 无法正确播放.")
                                else:
                                    # 将16-bit PCM字节数据转换为 int16 NumPy 数组
                                    audio_pcm_int16 = np.frombuffer(pcm_data_bytes, dtype=np.int16)
                                    
                                    # 首先将原始PCM转换为目标播放器期望的float32格式，此时仍是原始采样率
                                    audio_float32_original_sr = audio_pcm_int16.astype(np.float32) / 32768.0
                                    
                                    # 默认情况下，要播放的音频就是这个原始采样率的音频
                                    audio_to_play_float32 = audio_float32_original_sr 

                                    if wav_framerate != self.PLAYBACK_RATE:
                                        print(f"   ⚠️ [重采样] WAV文件采样率 ({wav_framerate}Hz) 与播放器预设采样率 ({self.PLAYBACK_RATE}Hz) 不同。正在尝试重采样...")
                                        try:
                                            # librosa.resample 的参数是 (y, orig_sr, target_sr)
                                            # y 是一个numpy数组，浮点型
                                            audio_to_play_float32 = librosa.resample(audio_float32_original_sr, 
                                                                                     orig_sr=wav_framerate, 
                                                                                     target_sr=self.PLAYBACK_RATE,
                                                                                     res_type='kaiser_best') # 明确指定重采样算法
                                            print(f"      ✅ [重采样] 音频已从 {wav_framerate}Hz 重采样到 {self.PLAYBACK_RATE}Hz.")
                                        except Exception as e_resample:
                                            print(f"      ❌ [重采样] 失败: {e_resample}.")
                                            print(f"         将尝试以原始采样率数据播放（可能导致播放速度不正确）。")
                                            # 如果重采样失败, audio_to_play_float32 保持为 audio_float32_original_sr
                                    
                                    # 如果WAV是立体声但我们只期望单声道，这里可以简单取一个声道，但这已由上面的channels检查阻止
                                    # if wav_channels == 2 and self.CHANNELS == 1:
                                    #    audio_to_play_float32 = audio_to_play_float32[::2] # 取左声道

                                    print(f"   [播放] 准备播放 {len(audio_to_play_float32)} 个采样点 (float32) 至设备 (配置为 {self.PLAYBACK_RATE}Hz)")
                                    
                                    time_playback_starts = time.time() # 记录播放开始时间
                                    if self.time_phrase_sent:
                                        latency_to_playback = time_playback_starts - self.time_phrase_sent
                                        print(f"⏱️⏱️ 端到端延迟 (发送 -> 开始播放): {latency_to_playback:.3f} 秒")
                                        self.time_phrase_sent = None # 重置，为下一段语音计时做准备

                                    audio_output.write(audio_to_play_float32)
                                    print(f"   [播放] 音频已发送到播放设备。")

                    except wave.Error as e_wave:
                        print(f"❌ 读取WAV数据失败: {e_wave}. 文件可能不是有效的WAV格式或者已损坏.")
                    except Exception as e_play:
                        print(f"❌ 播放音频时发生未知错误: {e_play}")
                        import traceback
                        traceback.print_exc()
                    
                    # print("⚠️  当前版本仅保存接收到的音频，未进行播放。") # 此行可以移除了

            except KeyboardInterrupt:
                print("\n⌨️ 收到键盘输入 - 客户端正在关闭")
            # ConnectionAbortedError 不再由我们主动抛出，而是依赖 _recv_all_data 返回 None
            except ConnectionResetError: # 可能由 self.socket.connect 或 self.record_callback中的send引发
                print("❌ 服务器连接被重置 - 客户端正在关闭 (外层捕获)")
            except socket.error as e: # 其他socket错误
                 print(f"Socket操作发生错误 (外层捕获): {e}")
            except Exception as e_main_loop:
                print(f"🔴 客户端主循环发生未知错误: {e_main_loop}")
                import traceback
                traceback.print_exc()
            finally:
                print("🔌 关闭socket连接...")
                try:
                    self.socket.shutdown(socket.SHUT_RDWR)
                except OSError as e:
                    is_not_connected_error = False
                    # Windows: [WinError 10057] A request to send or receive data was disallowed because the socket is not connected.
                    if hasattr(e, 'winerror') and e.winerror == 10057:
                        is_not_connected_error = True
                    # General socket error for not connected (errno can vary, e.g. ENOTCONN)
                    # Checking common string in error message as a fallback.
                    elif isinstance(e, socket.error) and "not connected" in str(e).lower():
                        is_not_connected_error = True
                    
                    if is_not_connected_error:
                        print("   Socket已断开或未连接，无需/无法shutdown。")
                    else:
                        print(f"   Shutdown时发生OSError: {e}")
                except Exception as e_shut:
                     print(f"   Shutdown时发生未知错误: {e_shut}")
                self.socket.close()
                print("✅ Socket连接已关闭。")

    def __volume_print_worker__(self):
        """ Event loop for worker to continually update the terminal volume meter"""
        last_volume_input = 0
        last_volume_output = 0
        print_sound(0, 0, blocks=10)
        while True:
            if abs(last_volume_input - self.volume_input) > 0.1 or abs(last_volume_output - self.volume_output) > 0.1:
                print_sound(self.volume_input, self.volume_output, blocks=10)
                last_volume_input = self.volume_input
                last_volume_output = self.volume_output
            if self.time_last_sent and time.time() - self.time_last_sent > self.PHRASE_TIME_LIMIT:
                self.volume_input = 0
            time.sleep(0.1)
    def __debug_worker__(self):
        """Background worker to handle debug statements"""
        print("Started background debug worker")
        while True:
            if not self.time_last_sent:
                # We can let the processor sleep more
                time.sleep(1)
                continue
            if not self.time_last_received:
                # Data has been sent, waiting on receiving
                time.sleep(0.05)
                continue
            if time.time() - self.time_last_received > self.time_flush_received:
                logging.debug("Last audio packet - time: %f",
                              self.time_last_received - self.time_last_sent)
                self.time_last_received = None
                self.time_first_received = None

if __name__ == "__main__":
    # 确保logs目录存在
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
        print(f"INFO: Created directory: {os.path.abspath(logs_dir)}")

    date_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(filename=f"{logs_dir}/{date_str}-output.log",
                        encoding='utf-8',
                        level=logging.DEBUG)
    # Hide cursor in terminal:
    print('\033[?25l', end="")
    # Start server
    client = AudioSocketClient()
    client.start('localhost', 4444)
    # Show cursor again:
    print('\033[?25h', end="")
