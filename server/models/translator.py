""" Translation module using various translation services """
import requests
import json
from typing import Optional
# from googletrans import Translator as GoogleTranslator # 旧的导入
from deep_translator import GoogleTranslator as DeepGoogleTranslator # 新的导入

class Translator:
    """翻译器类，支持多种翻译服务"""
    
    def __init__(self, service="google"):
        self.service = service
        if service == "google":
            try:
                # 初始化 deep-translator 的 GoogleTranslator
                self.google_translator = DeepGoogleTranslator(source='auto', target='en')
                print("✅ Deep Translator (Google) 初始化成功")
            except Exception as e:
                print(f"❌ Deep Translator (Google) 初始化失败: {e}")
                self.google_translator = None
        
    def translate_to_english(self, text: str) -> str:
        """将文本翻译为英文"""
        if not text or not text.strip():
            return ""
            
        try:
            if self.service == "google" and self.google_translator:
                # 使用 deep-translator进行翻译
                translated_text = self.google_translator.translate(text)
                return translated_text if translated_text else ""
            elif self.service == "baidu":
                return self._baidu_translate(text, "zh", "en")
            else:
                # 简单的本地翻译映射（作为备选）
                return self._simple_translate(text)
        except Exception as e:
            print(f"Translation error with service '{self.service}': {e}")
            return text  # 翻译失败时返回原文
    
    def _google_translate(self, text: str, source: str, target: str) -> str:
        """使用Google翻译API (旧的requests方法，保留作为参考或备用)"""
        # 使用免费的Google翻译接口
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            'client': 'gtx',
            'sl': source,
            'tl': target,
            'dt': 't',
            'q': text
        }
        
        try:
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                result = response.json()
                if result and len(result) > 0 and len(result[0]) > 0:
                    return result[0][0][0]
        except Exception as e:
            print(f"Google translate error: {e}")
            
        return text
    
    def _baidu_translate(self, text: str, source: str, target: str) -> str:
        """使用百度翻译API（需要配置API密钥）"""
        # 这里需要配置百度翻译的API密钥
        # 暂时返回原文
        return text
    
    def _simple_translate(self, text: str) -> str:
        """简单的本地翻译映射"""
        translation_dict = {
            "你好": "Hello",
            "你好。": "Hello.",
            "再见": "Goodbye",
            "再见。": "Goodbye.",
            "谢谢": "Thank you",
            "谢谢。": "Thank you.",
            "早上好": "Good morning",
            "早上好。": "Good morning.",
            "晚上好": "Good evening",
            "晚上好。": "Good evening.",
            "我爱你": "I love you",
            "我爱你。": "I love you.",
            "今天天气很好": "The weather is nice today",
            "今天天气很好。": "The weather is nice today.",
        }
        
        # 精确匹配
        if text in translation_dict:
            return translation_dict[text]
        
        # 模糊匹配
        for chinese, english in translation_dict.items():
            if chinese in text:
                return english
                
        return f"Hello, you said: {text}"  # 默认回复 