""" GPT-SoVITS Text to Speech with API Integration """
import requests
import json
import time
import io
import torch
import numpy as np
from typing import Optional
import urllib.parse

class GPTSoVITSTTSModel:
    """GPT-SoVITS TTS模型类，通过API调用进行语音合成"""
    
    def __init__(self, api_url="http://localhost:9872", callback_function=None):
        self.api_url = api_url.rstrip('/')
        self.callback_function = callback_function
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # 默认参考音频设置
        self.default_refer_wav_path = None
        self.default_prompt_text = None
        self.default_prompt_language = "zh"
        
        # 测试API连接
        self._test_connection()
        
        print(f"🚀 GPT-SoVITS TTS API connected: {self.api_url}")
        
    def _test_connection(self):
        """测试API连接"""
        try:
            response = requests.get(f"{self.api_url}/", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(f"API返回状态码: {response.status_code}")
        except Exception as e:
            print(f"❌ 无法连接到GPT-SoVITS API: {e}")
            print(f"请确保GPT-SoVITS服务正在运行在 {self.api_url}")
            raise e
    
    def synthesise(self, text: str, client_socket, 
                   refer_wav_path: Optional[str] = None,
                   prompt_text: Optional[str] = None,
                   prompt_language: str = "zh",
                   text_language: str = "zh"):
        """
        非阻塞语音合成
        
        Args:
            text: 要合成的文本
            client_socket: 客户端socket
            refer_wav_path: 参考音频路径
            prompt_text: 参考音频文本
            prompt_language: 参考音频语言
            text_language: 目标文本语言
        """
        if self.callback_function:
            # 异步处理
            audio = self.synthesise_blocking(
                text, refer_wav_path, prompt_text, 
                prompt_language, text_language
            )
            self.callback_function(audio, client_socket)
    
    def synthesise_blocking(self, text: str,
                           refer_wav_path: Optional[str] = None,
                           prompt_text: Optional[str] = None,
                           prompt_language: str = "zh",
                           text_language: str = "zh") -> torch.Tensor:
        """
        阻塞式语音合成
        
        Args:
            text: 要合成的文本
            refer_wav_path: 参考音频路径
            prompt_text: 参考音频文本  
            prompt_language: 参考音频语言
            text_language: 目标文本语言
            
        Returns:
            torch.Tensor: 合成的音频数据
        """
        start_time = time.time()
        
        try:
            # 使用默认参考音频（如果没有提供的话）
            if not refer_wav_path and self.default_refer_wav_path:
                refer_wav_path = self.default_refer_wav_path
                prompt_text = self.default_prompt_text
                prompt_language = self.default_prompt_language
            
            # 根据是否有参考音频选择不同的调用方式
            if refer_wav_path and prompt_text:
                # 使用POST方式，包含参考音频
                data = {
                    "refer_wav_path": refer_wav_path,
                    "prompt_text": prompt_text,
                    "prompt_language": prompt_language,
                    "text": text,
                    "text_language": text_language
                }
                
                response = requests.post(
                    f"{self.api_url}/",
                    json=data,
                    timeout=30
                )
            else:
                # 使用GET方式，不包含参考音频
                params = {
                    "text": text,
                    "text_language": text_language
                }
                
                # 构建URL参数
                query_string = urllib.parse.urlencode(params)
                url = f"{self.api_url}/?{query_string}"
                
                response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                # 获取音频数据
                audio_data = response.content
                
                if len(audio_data) == 0:
                    print("⚠️  收到空音频数据")
                    return torch.zeros(16000)
                
                # 将音频数据转换为torch tensor
                # GPT-SoVITS返回的是WAV格式音频
                try:
                    # 跳过WAV文件头（通常是44字节）
                    if len(audio_data) > 44:
                        audio_bytes = audio_data[44:]  # 跳过WAV头
                    else:
                        audio_bytes = audio_data
                    
                    # 转换为numpy数组
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                    
                    if len(audio_array) == 0:
                        print("⚠️  音频数组为空")
                        return torch.zeros(16000)
                    
                    # 转换为float32并归一化
                    audio_tensor = torch.from_numpy(audio_array.astype(np.float32) / 32768.0)
                    
                    end_time = time.time()
                    print(f"🔊 GPT-SoVITS合成完成: '{text}' 耗时: {end_time - start_time:.2f}秒")
                    print(f"   音频长度: {len(audio_tensor)} 样本")
                    
                    return audio_tensor
                    
                except Exception as e:
                    print(f"❌ 音频数据处理失败: {e}")
                    # 直接返回原始数据作为tensor
                    audio_array = np.frombuffer(audio_data, dtype=np.uint8)
                    audio_tensor = torch.from_numpy(audio_array.astype(np.float32))
                    return audio_tensor
                
            else:
                print(f"❌ GPT-SoVITS API错误: {response.status_code}")
                print(f"错误内容: {response.text}")
                # 返回空音频
                return torch.zeros(16000)  # 1秒的静音
                
        except Exception as e:
            print(f"❌ GPT-SoVITS合成失败: {e}")
            # 返回空音频
            return torch.zeros(16000)  # 1秒的静音
    
    def set_reference_audio(self, wav_path: str, text: str, language: str = "zh"):
        """
        设置默认参考音频
        
        Args:
            wav_path: 音频文件路径
            text: 音频对应文本
            language: 音频语言
        """
        self.default_refer_wav_path = wav_path
        self.default_prompt_text = text
        self.default_prompt_language = language
        print(f"✅ 设置默认参考音频: {wav_path}")
        print(f"   参考文本: {text}")
        print(f"   语言: {language}")
    
    def load_speaker_embeddings(self):
        """兼容性方法，GPT-SoVITS不需要预加载说话人嵌入"""
        print("✅ GPT-SoVITS不需要预加载说话人嵌入")
        pass 