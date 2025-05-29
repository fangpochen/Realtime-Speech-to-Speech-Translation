"""
GPT-SoVITS配置文件
用于配置语音合成的各种参数
"""

class GPTSoVITSConfig:
    """GPT-SoVITS配置类"""
    
    def __init__(self):
        # 基础配置
        self.api_url = "http://localhost:9872"
        self.ref_wav_path = "server/tts_wav/1.wav"  # 主参考音频路径
        self.ref_text_path = "server/tts_wav/1.txt" # 主参考音频的文本路径
        
        # /inference API 语音合成参数 (根据用户截图更新)
        # 核心采样参数
        self.top_k = 15               # int, Screenshot: 15 (API Default: 5)
        self.top_p = 1.0             # float, Screenshot: 1 (API Default: 1.0)
        self.temperature = 1.0       # float, Screenshot: 1 (API Default: 1.0)
        self.sample_steps = "64"      # str, Screenshot: 8 (API Default: "32")
        
        # 文本处理与切分
        self.text_split_method = "凑四句一切" # str, Screenshot: "凑四句一切" (API Default: "凑四句一切")
        self.fragment_interval = 0.5   # float, Screenshot: 0.5 (API Default: 0.3)

        # 语速与随机性控制
        self.speed_factor = 1.0      # float, Screenshot: 1 (API Default: 1.0)
        self.seed = 938619027.0    # float, Screenshot: 938619027 (API Default: -1.0)
        self.keep_random = True      # bool, Screenshot: True (API Default: True)

        # 参考模式与超分
        self.ref_text_free = False   # bool, (Not directly in screenshot's main params, assume API default or previous state)
        self.super_sampling = False  # bool, Screenshot: False (API Default: False)

        # 高级/性能参数
        self.batch_size = 50.0         # float, Screenshot: 25 (API Default: 20.0)
        self.split_bucket = True       # bool, Screenshot: True (API Default: True)
        self.parallel_infer = True     # bool, Screenshot: True (API Default: True)
        self.repetition_penalty = 1.35 # float, Screenshot: 1.35 (API Default: 1.35)
        
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
                # 特殊处理 sample_steps 以确保是字符串
                if key == "sample_steps" and not isinstance(value, str):
                    try:
                        value = str(int(value)) # 尝试从数字转换
                    except ValueError:
                        print(f"⚠️  sample_steps 值 '{value}' 无法转换为字符串，将保持原样。请确保它是有效字面量。")
                elif key == "seed" and not isinstance(value, float):
                    try:
                        value = float(value)
                    except ValueError:
                         print(f"⚠️  seed 值 '{value}' 无法转换为浮点数，将保持原样。")
                elif key == "top_k" and not isinstance(value, int):
                    try:
                        value = int(value)
                    except ValueError:
                         print(f"⚠️  top_k 值 '{value}' 无法转换为整数，将保持原样。")
                # 其他类型转换可以根据需要添加，例如 top_p, temperature, speed_factor 为 float

                setattr(self, key, value)
                print(f"✅ 更新配置: {key} = {value}")
            else:
                print(f"⚠️  未知配置项: {key}") 