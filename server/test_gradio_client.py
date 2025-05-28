#!/usr/bin/env python3
"""测试gradio_client连接GPT-SoVITS"""

import os
import sys
import traceback
from gradio_client import Client, file

def test_gradio_client(api_url="http://localhost:9872", text_to_synthesize="hello. im from American", language_of_text="英文"):
    """测试gradio_client连接"""
    
    print(f"🧪 测试Gradio Client连接: {api_url}")
    
    try:
        # 初始化客户端
        print(f"🔗 正在连接到Gradio服务... (SSL验证禁用)")
        client = Client(api_url, ssl_verify=False)
        print(f"✅ Gradio Client连接成功 (SSL验证已禁用)")
        
        # 使用用户提供的本地参考音频和文本
        ref_wav_path = os.path.abspath("server/tts_wav/1.wav")
        ref_text_path = os.path.abspath("server/tts_wav/1.txt")
        prompt_language_for_local_ref = "中文"

        print(f"📁 检查本地参考文件路径:")
        print(f"   音频文件: {ref_wav_path}")
        print(f"   文本文件: {ref_text_path}")
        
        if not os.path.exists(ref_wav_path):
            print(f"❌ 参考音频文件不存在: {ref_wav_path}")
            return False
            
        if not os.path.exists(ref_text_path):
            print(f"❌ 参考文本文件不存在: {ref_text_path}")
            return False
            
        print(f"✅ 本地文件检查通过")

        # 读取本地参考文本
        with open(ref_text_path, 'r', encoding='utf-8') as f:
            prompt_text_from_file = f.read().strip()
        print(f"📝 本地参考文本: '{prompt_text_from_file}'")
        
        # 测试API调用
        print(f"🔊 开始TTS合成测试...")
        print(f"📋 使用参数:")
        print(f"   - 参考音频: {ref_wav_path} (本地)")
        print(f"   - 参考文本: '{prompt_text_from_file}'")
        print(f"   - 参考语言: {prompt_language_for_local_ref}")
        print(f"   - 合成文本: {text_to_synthesize}")
        print(f"   - 合成语言: {language_of_text}")
        
        result = client.predict(
            ref_wav_path=file(ref_wav_path),
            prompt_text=prompt_text_from_file,
            prompt_language=prompt_language_for_local_ref,
            text=text_to_synthesize,
            text_language=language_of_text,
            how_to_cut="凑四句一切",
            top_k=15,
            top_p=1,
            temperature=1,
            ref_free=False,
            speed=1,
            if_freeze=False,
            inp_refs=None,
            sample_steps=8,
            if_sr=False,
            pause_second=0.3,
            api_name="/get_tts_wav"
        )
        
        print(f"✅ TTS合成成功!")
        print(f"📁 输出文件: {result}")
        
        if result and os.path.exists(result):
            file_size = os.path.getsize(result)
            print(f"📊 文件大小: {file_size} bytes")
            
            # 复制到当前目录
            import shutil
            output_filename = "test_gradio_output.wav"
            shutil.copy2(result, output_filename)
            print(f"✅ 音频已复制到: {os.path.abspath(output_filename)}")
            
            return True
        else:
            print(f"❌ 输出文件无效: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Gradio Client测试失败: {e}")
        print(f"🔍 详细错误信息:")
        traceback.print_exc()
        return False

def test_simple_connection(api_url="http://localhost:9872"):
    """测试简单连接"""
    try:
        print(f"🔗 测试简单连接... (SSL验证禁用)")
        client = Client(api_url, ssl_verify=False)
        print(f"✅ Gradio Client连接成功 (SSL验证已禁用)")
        
        # 尝试获取API信息
        print(f"📋 API信息:")
        # print(f"   - 端点: {client.endpoints}") # May cause issues if API is not fully responsive
        
        return True
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    api_url = "http://localhost:9872"
    # 使用用户指定的文本和语言进行合成
    text_to_synthesize_param = "hello. im from American"
    language_of_text_param = "英文"

    if len(sys.argv) > 1:
        api_url = sys.argv[1]
    # 命令行参数可以覆盖默认的合成文本和语言
    if len(sys.argv) > 2:
        text_to_synthesize_param = sys.argv[2]
    if len(sys.argv) > 3:
        language_of_text_param = sys.argv[3]
        
    print("=" * 50)
    print(f"Gradio Client TTS 测试 (本地参考, SSL禁用)")
    print(f"合成文本: \"{text_to_synthesize_param}\" ({language_of_text_param})")
    print("=" * 50)
    
    # 先测试简单连接
    if test_simple_connection(api_url):
        # 再测试完整功能
        success = test_gradio_client(api_url, text_to_synthesize_param, language_of_text_param)
        
        if success:
            print(f"\n🎉 Gradio Client TTS测试成功!")
            print(f"可以使用完整的GPT-SoVITS功能")
        else:
            print(f"\n❌ Gradio Client TTS测试失败!")
    else:
        print(f"\n❌ 无法连接到GPT-SoVITS服务")
    
    print("=" * 50) 