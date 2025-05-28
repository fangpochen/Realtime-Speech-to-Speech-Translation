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
from gradio_client import Client, file
from models.speech_recognition_funasr import FunASRSpeechRecognitionModel
from models.translator import Translator
from gpt_sovits_config import GPTSoVITSConfig

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
        print(f"🎤 识别结果: '{packet}'")
        
        if not packet or not packet.strip():
            print("⚠️  识别结果为空，跳过翻译")
            return
        
        # 翻译为英文
        print(f"🔄 开始翻译...")
        translated_text = self.translator.translate_to_english(packet)
        print(f"🌍 翻译结果: '{translated_text}'")
        
        if translated_text and translated_text.strip():
            print(f"🔊 开始GPT-SoVITS语音合成...")
            # 使用GPT-SoVITS合成英文语音
            audio_data = self.gpt_sovits_synthesize(translated_text, "en")
            if audio_data:
                self.stream_audio_to_client(audio_data, client_socket)
        else:
            print("⚠️  翻译结果为空，跳过语音合成")

    def gpt_sovits_synthesize(self, text: str, text_language: str = "en"):
        """调用GPT-SoVITS /inference API进行语音合成"""
        if not self.gpt_sovits_client:
            print("❌ GPT-SoVITS客户端未初始化，无法进行语音合成。")
            return None
        try:
            print(f"🔊 开始GPT-SoVITS合成 (API: /inference): '{text}'")
            
            # 根据目标语言设置参考音频语言
            prompt_language_literal = self.gpt_config.get_language("zh")  # 参考音频是中文
            text_language_literal = self.gpt_config.get_language(text_language) # 目标合成语言
            
            # 从 self.gpt_config 获取参数，如果属性不存在则使用默认值
            # 这些默认值应与 /inference API 的默认值或您的期望相匹配
            ref_audio_path_to_use = self.ref_wav_path # 旧的属性名，但路径应指向主参考
            prompt_text_to_use = self.ref_text

            top_k_to_use = getattr(self.gpt_config, 'top_k', 5) 
            top_p_to_use = getattr(self.gpt_config, 'top_p', 1.0)
            temperature_to_use = getattr(self.gpt_config, 'temperature', 1.0)
            
            # 映射旧参数名到新参数名，并从config获取或使用默认值
            text_split_method_to_use = getattr(self.gpt_config, 'text_split_method', getattr(self.gpt_config, 'how_to_cut', "凑四句一切"))
            speed_factor_to_use = getattr(self.gpt_config, 'speed_factor', getattr(self.gpt_config, 'speed', 1.0))
            ref_text_free_to_use = getattr(self.gpt_config, 'ref_text_free', getattr(self.gpt_config, 'ref_free', False))
            fragment_interval_to_use = getattr(self.gpt_config, 'fragment_interval', getattr(self.gpt_config, 'pause_second', 0.3))
            
            # 新 /inference API 特定参数
            seed_to_use = float(getattr(self.gpt_config, 'seed', -1.0)) # 确保是float
            keep_random_to_use = getattr(self.gpt_config, 'keep_random', True)
            sample_steps_to_use = str(getattr(self.gpt_config, 'sample_steps', "32")) # 确保是str

            # 其他 /inference API 参数 (当前从 config 获取或使用硬编码的 API 默认值)
            # 用户可能希望将这些也加入 GPTSoVITSConfig
            batch_size_to_use = getattr(self.gpt_config, 'batch_size', 20.0)
            split_bucket_to_use = getattr(self.gpt_config, 'split_bucket', True)
            parallel_infer_to_use = getattr(self.gpt_config, 'parallel_infer', True)
            repetition_penalty_to_use = getattr(self.gpt_config, 'repetition_penalty', 1.35)
            super_sampling_to_use = getattr(self.gpt_config, 'super_sampling', getattr(self.gpt_config, 'if_sr', False))


            print(f"   [GPT-SoVITS Params] Seed: {seed_to_use}, KeepRandom: {keep_random_to_use}, Temp: {temperature_to_use}")
            print(f"   [GPT-SoVITS Params] TopK: {top_k_to_use}, TopP: {top_p_to_use}, SampleSteps: {sample_steps_to_use}")

            # 调用GPT-SoVITS /inference API
            result_tuple = self.gpt_sovits_client.predict(
                text=text,
                text_lang=text_language_literal, # 使用转换后的字面量
                ref_audio_path=file(ref_audio_path_to_use), # 参数名更改
                aux_ref_audio_paths=[], # 必需参数，默认为空列表
                prompt_text=prompt_text_to_use,
                prompt_lang=prompt_language_literal, # 使用转换后的字面量
                top_k=top_k_to_use,
                top_p=top_p_to_use,
                temperature=temperature_to_use,
                text_split_method=text_split_method_to_use, # 参数名更改
                speed_factor=speed_factor_to_use, # 参数名更改
                ref_text_free=ref_text_free_to_use, # 新参数 (或旧参数映射)
                split_bucket=split_bucket_to_use, # 新参数
                fragment_interval=fragment_interval_to_use, # 新参数 (或旧参数映射)
                seed=seed_to_use, # 新参数
                keep_random=keep_random_to_use, # 新参数
                parallel_infer=parallel_infer_to_use, # 新参数
                repetition_penalty=repetition_penalty_to_use, # 新参数
                sample_steps=sample_steps_to_use, # 参数类型可能变化，确保是str
                super_sampling=super_sampling_to_use, # 新参数 (或旧参数映射)
                batch_size=batch_size_to_use, # 新参数
                api_name="/inference" # 明确指定新的API端点
            )
            
            # 处理 /inference API 的返回结果 (filepath, seed_float)
            if isinstance(result_tuple, tuple) and len(result_tuple) == 2:
                output_audio_path, returned_seed = result_tuple
                print(f"   [GPT-SoVITS API] Returned audio path: {output_audio_path}, Returned seed: {returned_seed}")
                if output_audio_path and os.path.exists(output_audio_path):
                    with open(output_audio_path, 'rb') as f:
                        audio_data = f.read()
                    print(f"🔊 GPT-SoVITS合成完成: '{text}' 音频大小: {len(audio_data)} bytes, 使用Seed: {returned_seed}")
                    # os.remove(output_audio_path) # 可选：删除服务端的临时文件
                    return audio_data
                else:
                    print(f"❌ GPT-SoVITS /inference API返回无效音频路径: {output_audio_path}")
                    return None
            else:
                print(f"❌ GPT-SoVITS /inference API返回结果格式不符合预期 (应为元组): {result_tuple}")
                return None
                
        except Exception as e:
            print(f"❌ GPT-SoVITS合成失败 (调用/inference): {e}")
            import traceback
            traceback.print_exc() # 打印更详细的错误堆栈
            return None

    def stream_audio_to_client(self, audio_data: bytes, client_socket):
        """将音频数据发送到客户端"""
        try:
            if client_socket and hasattr(client_socket, 'sendall'):
                # 跳过WAV文件头，直接发送音频数据
                if len(audio_data) > 44:
                    audio_bytes = audio_data[44:]  # 跳过WAV头
                else:
                    audio_bytes = audio_data
                
                client_socket.sendall(audio_bytes)
                print(f"✅ GPT-SoVITS音频已发送到客户端，大小: {len(audio_bytes)} bytes")
            else:
                print("⚠️  客户端连接已断开，无法发送音频")
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            print(f"❌ 发送音频失败: {e}")
            if client_socket in self.read_list:
                self.read_list.remove(client_socket)

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
                                print(f"📥 收到音频数据: {len(data)} bytes")
                                self.data_queue.put((s, data))
                            else:
                                self.read_list.remove(s)
                                print("Disconnection from", address)
                        except ConnectionResetError:
                            self.read_list.remove(s)
                            print("Client crashed from", address)
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