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
        """调用GPT-SoVITS API进行语音合成"""
        if not self.gpt_sovits_client:
            print("❌ GPT-SoVITS客户端未初始化，无法进行语音合成。")
            return None
        try:
            print(f"🔊 开始GPT-SoVITS合成: '{text}'")
            
            # 根据目标语言设置参考音频语言
            prompt_language = self.gpt_config.get_language("zh")  # 参考音频是中文
            target_language = self.gpt_config.get_language(text_language)
            
            # 调用GPT-SoVITS API
            result = self.gpt_sovits_client.predict(
                ref_wav_path=file(self.ref_wav_path),
                prompt_text=self.ref_text,
                prompt_language=prompt_language,
                text=text,
                text_language=target_language,
                how_to_cut=self.gpt_config.how_to_cut,
                top_k=self.gpt_config.top_k,
                top_p=self.gpt_config.top_p,
                temperature=self.gpt_config.temperature,
                ref_free=self.gpt_config.ref_free,
                speed=self.gpt_config.speed,
                if_freeze=self.gpt_config.if_freeze,
                inp_refs=None,
                sample_steps=self.gpt_config.sample_steps,
                if_sr=self.gpt_config.if_sr,
                pause_second=self.gpt_config.pause_second,
                api_name="/get_tts_wav"
            )
            
            # 读取生成的音频文件
            if result and os.path.exists(result):
                with open(result, 'rb') as f:
                    audio_data = f.read()
                print(f"🔊 GPT-SoVITS合成完成: '{text}' 音频大小: {len(audio_data)} bytes")
                return audio_data
            else:
                print(f"❌ GPT-SoVITS API返回无效结果: {result}")
                return None
                
        except Exception as e:
            print(f"❌ GPT-SoVITS合成失败: {e}")
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
        """获取当前GPT-SoVITS配置"""
        config_dict = {
            'api_url': self.gpt_config.api_url,
            'ref_wav_path': self.gpt_config.ref_wav_path,
            'ref_text_path': self.gpt_config.ref_text_path,
            'how_to_cut': self.gpt_config.how_to_cut,
            'top_k': self.gpt_config.top_k,
            'top_p': self.gpt_config.top_p,
            'temperature': self.gpt_config.temperature,
            'ref_free': self.gpt_config.ref_free,
            'speed': self.gpt_config.speed,
            'if_freeze': self.gpt_config.if_freeze,
            'sample_steps': self.gpt_config.sample_steps,
            'if_sr': self.gpt_config.if_sr,
            'pause_second': self.gpt_config.pause_second
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