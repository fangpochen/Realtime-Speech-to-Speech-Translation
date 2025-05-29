""" Server for real-time translation and voice synthesization using FunASR """
from typing import Dict
from queue import Queue
import select
import socket
import pyaudio
import torch
import requests
import numpy as np
import urllib.parse
import os
import struct
from gradio_client import Client, file
from models.speech_recognition_funasr import FunASRSpeechRecognitionModel
from models.translator import Translator
from gpt_sovits_config import GPTSoVITSConfig
import time

class AudioSocketServerFunASR:
    """ Class that handles real-time translation and voice synthesization using FunASR
        Socket input -> FunASR -> text -> TextToSpeech -> Socket output
    """
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 4096
    PORT = 4444
    # Number of unaccepted connections before server refuses new connections.
    BACKLOG = 5
    
    def __init__(self, funasr_model="paraformer-zh", gpt_sovits_api="http://localhost:9872"):
        self.audio = pyaudio.PyAudio()
        self.serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Let kernel know we want to reuse the same port for restarting the server
        self.serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # TODO: For multiple concurrent users we will need more queues
        self.data_queue : Queue = Queue()

        # Initialize the FunASR transcriber model
        self.transcriber = FunASRSpeechRecognitionModel(
            model_name=funasr_model,
            data_queue=self.data_queue,
            generation_callback=self.handle_generation,
            final_callback=self.handle_transcription
        )
        
        # 初始化GPT-SoVITS配置
        self.gpt_config = GPTSoVITSConfig()
        self.gpt_config.api_url = gpt_sovits_api.rstrip('/')
        print(f"🚀 连接到GPT-SoVITS API: {self.gpt_config.api_url}")

        # 应用来自 test_gradio_client.py 的参数
        self.gpt_config.top_k = 20
        self.gpt_config.top_p = 1.0
        self.gpt_config.temperature = 1.0
        self.gpt_config.text_split_method = "凑四句一切"
        self.gpt_config.speed_factor = 1.0
        self.gpt_config.seed = -1.0
        self.gpt_config.keep_random = True
        self.gpt_config.sample_steps = "8" # 更新以匹配 test_gradio_client.py

        self.gpt_config.batch_size = 25.0 # 更新以匹配 test_gradio_client.py
        self.gpt_config.ref_text_free = False # test_gradio_client.py 中为 False
        self.gpt_config.split_bucket = True # test_gradio_client.py 中为 True
        self.gpt_config.fragment_interval = 0.3 # test_gradio_client.py 中为 0.3
        self.gpt_config.parallel_infer = True # test_gradio_client.py 中为 True
        self.gpt_config.repetition_penalty = 1.35 # test_gradio_client.py 中为 1.35
        self.gpt_config.super_sampling = False # test_gradio_client.py 中为 False
        
        print("ℹ️ GPT-SoVITS 配置已更新为来自 test_gradio_client.py 的参数。")
        
        # 初始化Gradio客户端
        try:
            self.gpt_sovits_client = Client(self.gpt_config.api_url, ssl_verify=False)
            print("✅ Gradio Client 初始化成功 (SSL验证已禁用)")
        except Exception as e:
            print(f"❌ Gradio Client 初始化失败: {e}")
            print("   请检查Gradio服务是否正在运行，以及SSL_CERT_FILE环境变量是否正确设置（如果未使用ssl_verify=False）。")
            # 可以选择在这里抛出异常或允许服务器继续运行但TTS功能受限
            self.gpt_sovits_client = None # 标记客户端不可用
        
        # 参考音频配置
        self.ref_wav_path = os.path.abspath(self.gpt_config.ref_wav_path)
        self.ref_text_path = os.path.abspath(self.gpt_config.ref_text_path)
        
        # 读取参考文本
        try:
            with open(self.ref_text_path, 'r', encoding='utf-8') as f:
                self.ref_text = f.read().strip()
            print(f"📝 参考文本: {self.ref_text}")
        except Exception as e:
            print(f"❌ 读取参考文本失败: {e}")
            self.ref_text = "可以可以可以。那我先上去。你等下就到那个办公室里去哈"
        
        # 初始化翻译器
        self.translator = Translator(service="google")  # 使用Google翻译
        
        self.read_list = []

    def __del__(self):
        self.audio.terminate()
        self.transcriber.stop()
        self.serversocket.shutdown()
        self.serversocket.close()
        
    def handle_generation(self, packet: Dict):
        """ Placeholder function for transcription"""
        pass
        
    def handle_transcription(self, packet: str, client_socket):
        """ Callback function to put finalized transcriptions into TTS"""
        asr_end_time = time.time()
        print(f"🎤 [{asr_end_time:.3f}] 识别结果: '{packet}'")
        
        if not packet or not packet.strip():
            print("⚠️  识别结果为空，跳过翻译")
            return
        
        # 翻译为英文
        translation_start_time = time.time()
        print(f"🔄 [{translation_start_time:.3f}] 开始翻译...")
        translated_text = self.translator.translate_to_english(packet)
        translation_end_time = time.time()
        print(f"🌍 [{translation_end_time:.3f}] 翻译结果: '{translated_text}' (耗时: {translation_end_time - translation_start_time:.3f}s)")
        
        if translated_text and translated_text.strip():
            tts_start_time = time.time()
            print(f"🔊 [{tts_start_time:.3f}] 开始GPT-SoVITS语音合成...")
            # 使用GPT-SoVITS合成英文语音
            audio_data, original_text_for_filename = self.gpt_sovits_synthesize(translated_text, "en")
            tts_end_time = time.time()
            if audio_data:
                print(f"合成完成，准备发送 (TTS总耗时: {tts_end_time - tts_start_time:.3f}s)")
                self.stream_audio_to_client(audio_data, client_socket, original_text_for_filename)
            else:
                print(f"⚠️ [{tts_end_time:.3f}] 语音合成失败或未返回数据 (TTS尝试耗时: {tts_end_time - tts_start_time:.3f}s)")
        else:
            print("⚠️  翻译结果为空，跳过语音合成")

    def gpt_sovits_synthesize(self, text: str, text_language: str = "en"):
        """调用GPT-SoVITS /inference API进行语音合成"""
        if not self.gpt_sovits_client:
            print("❌ GPT-SoVITS客户端未初始化，无法进行语音合成。")
            return None, None
        try:
            synthesis_api_call_start_time = time.time()
            print(f"🔊 [{synthesis_api_call_start_time:.3f}] 开始GPT-SoVITS合成 (API: /inference): '{text}'")
            
            prompt_language_literal = self.gpt_config.get_language("zh")
            text_language_literal = self.gpt_config.get_language(text_language)
            
            ref_audio_path_to_use = self.ref_wav_path
            
            # 从 ref_text_path 读取参考文本
            prompt_text_to_use = ""
            if os.path.exists(self.ref_text_path):
                with open(self.ref_text_path, 'r', encoding='utf-8') as f:
                    prompt_text_to_use = f.read().strip()
            else:
                print(f"⚠️ 参考文本文件未找到: {self.ref_text_path}, 使用空字符串。")

            params_to_api = {
                "text": text,
                "text_lang": text_language_literal,
                "ref_audio_path": file(ref_audio_path_to_use),
                "aux_ref_audio_paths": [], # 根据API定义，如果不需要则为空列表
                "prompt_text": prompt_text_to_use,
                "prompt_lang": prompt_language_literal,
                "top_k": self.gpt_config.top_k,
                "top_p": self.gpt_config.top_p,
                "temperature": self.gpt_config.temperature,
                "text_split_method": self.gpt_config.text_split_method,
                "batch_size": self.gpt_config.batch_size,
                "speed_factor": self.gpt_config.speed_factor,
                "ref_text_free": self.gpt_config.ref_text_free,
                "split_bucket": self.gpt_config.split_bucket,
                "fragment_interval": self.gpt_config.fragment_interval,
                "seed": self.gpt_config.seed,
                "keep_random": self.gpt_config.keep_random,
                "parallel_infer": self.gpt_config.parallel_infer,
                "repetition_penalty": self.gpt_config.repetition_penalty,
                "sample_steps": self.gpt_config.sample_steps,
                "super_sampling": self.gpt_config.super_sampling,
                "api_name": "/inference"
            }
            
            print("   [GPT-SoVITS Params] Preparing to call predict with:")
            # for k, v in params_to_api.items():
            #     if k == "ref_audio_path":
            #         print(f"     {k}: {ref_audio_path_to_use} (gradio.file object)")
            #     else:
            #         print(f"     {k}: {v}")
            # 打印关键参数
            print(f"     Text: '{params_to_api['text']}' ({params_to_api['text_lang']})")
            print(f"     Ref Audio: {ref_audio_path_to_use}")
            print(f"     Prompt Text: '{params_to_api['prompt_text']}' ({params_to_api['prompt_lang']})")
            print(f"     Seed: {params_to_api['seed']}, Keep Random: {params_to_api['keep_random']}")
            print(f"     Sample Steps: {params_to_api['sample_steps']}, Temperature: {params_to_api['temperature']}")


            predict_call_start_time = time.time()
            result_tuple = self.gpt_sovits_client.predict(**params_to_api)
            predict_call_end_time = time.time()
            print(f"   [GPT-SoVITS API Call] predict耗时: {predict_call_end_time - predict_call_start_time:.3f}s")

            if isinstance(result_tuple, tuple) and len(result_tuple) == 2:
                output_audio_path, returned_seed = result_tuple
                print(f"   [GPT-SoVITS API] Returned audio path: {output_audio_path}, Returned seed: {returned_seed}")
                if output_audio_path and os.path.exists(output_audio_path):
                    try:
                        save_dir = os.path.join(os.path.dirname(__file__), "server_outputs", "sovits_raw_outputs")
                        os.makedirs(save_dir, exist_ok=True)
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        safe_text_suffix = "".join(filter(str.isalnum, text[:20]))
                        filename = f"{timestamp}_{str(returned_seed).replace('.', '')}_{safe_text_suffix}.wav"
                        raw_output_filepath = os.path.join(save_dir, filename)
                        
                        import shutil
                        shutil.copy2(output_audio_path, raw_output_filepath)
                        print(f"   [GPT-SoVITS API] Raw output saved to: {raw_output_filepath}")
                        
                        # 读取保存的或原始的API输出音频文件内容以供发送
                        with open(raw_output_filepath, 'rb') as f_audio:
                            audio_data_for_client = f_audio.read()
                        synthesis_api_call_end_time = time.time()
                        print(f"   [GPT-SoVITS API] 整个合成函数耗时: {synthesis_api_call_end_time - synthesis_api_call_start_time:.3f}s")
                        return audio_data_for_client, text # 返回读取到的音频数据和原始文本
                    except Exception as e_save:
                        print(f"❌ 保存或读取SoVITS原始输出音频失败: {e_save}")
                        return None, None
                else:
                    print("❌ SoVITS API未返回有效音频路径或文件不存在。")
                    return None, None
            else:
                print(f"❌ SoVITS API返回结果格式不符合预期: {result_tuple}")
                return None, None
        except Exception as e:
            print(f"❌ 调用GPT-SoVITS API失败: {e}")
            import traceback
            traceback.print_exc()
            synthesis_api_call_failed_time = time.time()
            print(f"   [GPT-SoVITS API] 合成函数失败耗时: {synthesis_api_call_failed_time - synthesis_api_call_start_time:.3f}s")
            return None, None # 返回 None, None 表示失败

    def stream_audio_to_client(self, audio_data: bytes, client_socket, original_text="unknown"):
        """将音频数据(前缀长度头)发送到客户端，并在发送前保存一份以供调试"""
        try:
            if client_socket and hasattr(client_socket, 'sendall'):
                send_start_time = time.time()
                audio_bytes_to_send = audio_data

                # 调试：在发送前保存一份完整的WAV文件
                try:
                    save_dir = os.path.join(os.path.dirname(__file__), "server_outputs", "funasr_sent_audio")
                    os.makedirs(save_dir, exist_ok=True)
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    safe_text_suffix = "".join(filter(str.isalnum, original_text[:20])) if original_text else "audio"
                    debug_filename = f"{timestamp}_{safe_text_suffix}_sent_to_client.wav"
                    debug_filepath = os.path.join(save_dir, debug_filename)
                    with open(debug_filepath, 'wb') as f_debug:
                        f_debug.write(audio_bytes_to_send)
                    print(f"🔍 [调试] 即将发送的音频已保存到: {debug_filepath}, 大小: {len(audio_bytes_to_send)} bytes")
                except Exception as e_save:
                    print(f"⚠️ [调试] 保存发送前音频失败: {e_save}")

                # 1. 准备长度头 (8字节，网络字节序，无符号长整型)
                data_len = len(audio_bytes_to_send)
                header = struct.pack("!Q", data_len) # Q is for unsigned long long (8 bytes)

                # 2. 发送长度头
                send_header_start_time = time.time()
                client_socket.sendall(header)
                send_header_end_time = time.time()
                print(f"✉️  [{send_header_end_time:.3f}] 已发送数据长度头部: {data_len} bytes (头部本身 {len(header)} bytes, 发送耗时: {send_header_end_time - send_header_start_time:.3f}s)")

                # 3. 发送实际音频数据
                send_data_start_time = time.time()
                client_socket.sendall(audio_bytes_to_send)
                send_data_end_time = time.time()
                print(f"✅ [{send_data_end_time:.3f}] 音频数据已发送到客户端 (实际大小: {data_len} bytes, 发送耗时: {send_data_end_time - send_data_start_time:.3f}s)")
                print(f"   [Total Send Time] 总发送耗时: {send_data_end_time - send_start_time:.3f}s")
                
                # 移除 client_socket.shutdown(socket.SHUT_WR)
                # print("ℹ️  保持连接开放，以便发送更多音频。") # 可选的日志

            else:
                print("⚠️  客户端连接已断开或无效，无法发送音频")
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            # BrokenPipeError (errno 32) 可能会在客户端已关闭连接时发生
            # ConnectionResetError (errno 104) 也表示连接问题
            print(f"❌ 发送音频数据失败 (连接可能已由客户端关闭): {e}")
            # 如果发送失败，可能需要从 read_list 中移除此socket，避免select错误
            if client_socket in self.read_list:
                print(f"ℹ️  从监听列表中移除故障socket: {client_socket.getpeername() if hasattr(client_socket, 'getpeername') else client_socket}")
                self.read_list.remove(client_socket)
                try:
                    client_socket.close() # 彻底关闭这个出错的socket
                except Exception as e_close:
                    print(f"⚠️ 关闭故障socket时发生错误: {e_close}")
            # 不再向上抛出，允许服务器继续为其他客户端服务

    def start(self):
        """ Starts the server"""
        self.transcriber.start(16000, 2)
        print(f"🚀 GPT-SoVITS Translation Server listening on port {self.PORT}")
        print(f"📡 Connected to GPT-SoVITS API: {self.gpt_config.api_url}")
        self.serversocket.bind(('', self.PORT))
        self.serversocket.listen(self.BACKLOG)
        # Contains all of the socket connections
        self.read_list = [self.serversocket]

        try:
            while True:
                readable, _, _ = select.select(self.read_list, [], [])
                for s in readable:
                    if s is self.serversocket:
                        (clientsocket, address) = self.serversocket.accept()
                        self.read_list.append(clientsocket)
                        print("Connection from", address)
                    else:
                        try:
                            data = s.recv(4096)
                            if data:
                                self.data_queue.put((s, data))
                            else:
                                print(f"ℹ️  客户端 {address} 断开连接 (recv返回空数据)") # 增加地址信息
                                self.read_list.remove(s)
                                # print("Disconnection from", address) # 此行重复
                        except ConnectionResetError:
                            print(f"❌ 客户端 {address} 连接被重置") # 增加地址信息
                            self.read_list.remove(s)
                            # print("Client crashed from", address) # 此行重复
        except KeyboardInterrupt:
            pass
        print("Performing server cleanup")
        self.audio.terminate()
        self.transcriber.stop()
        self.serversocket.shutdown(socket.SHUT_RDWR)
        self.serversocket.close()
        print("Sockets cleaned up")

    def update_gpt_sovits_config(self, **kwargs):
        """更新GPT-SoVITS配置参数"""
        self.gpt_config.update_config(**kwargs)
        
    def get_gpt_sovits_config(self):
        """获取当前GPT-SoVITS配置 (适配 /inference API)"""
        config_dict = {
            'api_url': self.gpt_config.api_url,
            'ref_wav_path': self.gpt_config.ref_wav_path,
            'ref_text_path': self.gpt_config.ref_text_path,
            
            # /inference API 参数
            'top_k': self.gpt_config.top_k,
            'top_p': self.gpt_config.top_p,
            'temperature': self.gpt_config.temperature,
            'sample_steps': self.gpt_config.sample_steps, # str
            
            'text_split_method': self.gpt_config.text_split_method,
            'fragment_interval': self.gpt_config.fragment_interval,
            
            'speed_factor': self.gpt_config.speed_factor,
            'seed': self.gpt_config.seed, # float
            'keep_random': self.gpt_config.keep_random, # bool
            
            'ref_text_free': self.gpt_config.ref_text_free,
            'super_sampling': self.gpt_config.super_sampling,
            
            # 高级/性能参数
            'batch_size': self.gpt_config.batch_size,
            'split_bucket': self.gpt_config.split_bucket,
            'parallel_infer': self.gpt_config.parallel_infer,
            'repetition_penalty': self.gpt_config.repetition_penalty,
        }
        return config_dict

if __name__ == "__main__":
    import sys
    
    gpt_sovits_api = "http://localhost:9872"
    if len(sys.argv) > 1:
        gpt_sovits_api = sys.argv[1]
    
    server = AudioSocketServerFunASR(
        funasr_model="paraformer-zh",
        gpt_sovits_api=gpt_sovits_api
    )
    server.start() 