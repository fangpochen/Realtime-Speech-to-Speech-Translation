"""
GPT-SoVITS配置文件
用于配置语音合成的各种参数
"""

class GPTSoVITSConfig:
    """GPT-SoVITS配置类"""
    
    def __init__(self):
        # 基础配置
        self.api_url = "http://localhost:9872"
        self.ref_wav_path = "server/tts_wav/1.wav"
        self.ref_text_path = "server/tts_wav/1.txt"
        
        # 语音合成参数
        self.how_to_cut = "凑四句一切"  # 可选: '不切', '凑四句一切', '凑50字一切', '按中文句号。切', '按英文句号.切', '按标点符号切'
        self.top_k = 15
        self.top_p = 1
        self.temperature = 1
        self.ref_free = False
        self.speed = 1
        self.if_freeze = False
        self.sample_steps = "8"  # 可选: '4', '8', '16', '32'
        self.if_sr = False
        self.pause_second = 0.3
        
        # 语言映射
        self.language_mapping = {
            "en": "英文",
            "zh": "中文", 
            "ja": "日文",
            "yue": "粤语",
            "ko": "韩文"
        }
    
    def get_language(self, lang_code: str) -> str:
        """获取语言名称"""
        return self.language_mapping.get(lang_code, "中文")
    
    def update_config(self, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                print(f"✅ 更新配置: {key} = {value}")
            else:
                print(f"⚠️  未知配置项: {key}") 