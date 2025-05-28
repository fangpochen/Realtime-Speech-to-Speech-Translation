#!/usr/bin/env python3
"""测试GPT-SoVITS API连接和功能"""
import requests
import json
import time

def test_gpt_sovits_api(api_url="http://localhost:9872"):
    """测试GPT-SoVITS API"""
    
    print(f"🧪 测试GPT-SoVITS API: {api_url}")
    
    # 1. 测试API连接
    try:
        response = requests.get(f"{api_url}/", timeout=5)
        print(f"✅ API连接成功，状态码: {response.status_code}")
    except Exception as e:
        print(f"❌ API连接失败: {e}")
        return False
    
    # 2. 测试简单的TTS请求 (GET方式)
    try:
        print(f"🔊 测试TTS合成 (GET方式)...")
        start_time = time.time()
        
        # 使用GET方式，参数在URL中
        test_url = f"{api_url}/?text=Hello, this is a test.&text_language=en"
        
        response = requests.get(test_url, timeout=30)
        
        end_time = time.time()
        
        if response.status_code == 200:
            audio_size = len(response.content)
            print(f"✅ TTS合成成功!")
            print(f"   - 耗时: {end_time - start_time:.2f}秒")
            print(f"   - 音频大小: {audio_size} bytes")
            
            # 保存测试音频
            with open("test_output_get.wav", "wb") as f:
                f.write(response.content)
            print(f"   - 音频已保存为: test_output_get.wav")
            
            return True
        else:
            print(f"❌ TTS合成失败，状态码: {response.status_code}")
            print(f"   错误信息: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ TTS测试失败: {e}")
        return False

def test_post_request(api_url="http://localhost:9872"):
    """测试POST请求方式"""
    
    print(f"\n🔊 测试TTS合成 (POST方式)...")
    
    test_data = {
        "text": "你好，这是一个测试。",
        "text_language": "zh"
    }
    
    try:
        start_time = time.time()
        
        response = requests.post(
            f"{api_url}/",
            json=test_data,
            timeout=30
        )
        
        end_time = time.time()
        
        if response.status_code == 200:
            audio_size = len(response.content)
            print(f"✅ POST TTS合成成功!")
            print(f"   - 耗时: {end_time - start_time:.2f}秒")
            print(f"   - 音频大小: {audio_size} bytes")
            
            # 保存测试音频
            with open("test_output_post.wav", "wb") as f:
                f.write(response.content)
            print(f"   - 音频已保存为: test_output_post.wav")
            
            return True
        else:
            print(f"❌ POST TTS合成失败，状态码: {response.status_code}")
            print(f"   错误信息: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ POST TTS测试失败: {e}")
        return False

def test_with_reference_audio(api_url="http://localhost:9872"):
    """测试带参考音频的TTS"""
    
    print(f"\n🎤 测试带参考音频的TTS...")
    
    test_data = {
        "refer_wav_path": "example_reference.wav",  # 需要替换为实际的参考音频路径
        "prompt_text": "这是参考音频的文本",
        "prompt_language": "zh",
        "text": "你好，这是一个带参考音频的测试。",
        "text_language": "zh"
    }
    
    try:
        response = requests.post(
            f"{api_url}/",
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            print(f"✅ 带参考音频的TTS合成成功!")
            with open("test_output_with_ref.wav", "wb") as f:
                f.write(response.content)
            print(f"   - 音频已保存为: test_output_with_ref.wav")
            return True
        else:
            print(f"❌ 带参考音频的TTS失败: {response.status_code}")
            print(f"   错误信息: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 带参考音频的TTS测试失败: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    # 从命令行参数获取API地址
    api_url = "http://localhost:9872"
    if len(sys.argv) > 1:
        api_url = sys.argv[1]
    
    print("=" * 50)
    print("GPT-SoVITS API 测试工具")
    print("=" * 50)
    
    # 基础API测试
    success = test_gpt_sovits_api(api_url)
    
    if success:
        # 测试POST方式
        test_post_request(api_url)
        
        print(f"\n🎉 GPT-SoVITS API测试通过!")
        print(f"可以启动集成服务器:")
        print(f"python server_gpt_sovits.py {api_url}")
    else:
        print(f"\n❌ GPT-SoVITS API测试失败!")
        print(f"请检查:")
        print(f"1. GPT-SoVITS是否正在运行")
        print(f"2. API地址是否正确: {api_url}")
        print(f"3. 防火墙设置")
        print(f"4. 是否需要设置默认参考音频")
    
    print("=" * 50) 