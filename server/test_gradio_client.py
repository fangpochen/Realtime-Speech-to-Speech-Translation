#!/usr/bin/env python3
"""测试gradio_client连接GPT-SoVITS Inference API"""

import os
import sys
import traceback
import random
from gradio_client import Client, file

def test_gradio_client(api_url="http://localhost:9872",
                       text_to_synthesize="hello. im from American",
                       text_lang_param="英文",
                       # 以下参数将从固定路径加载，不再作为函数参数
                       # ref_audio_path_param=os.path.abspath("server/tts_wav/1.wav"), 
                       # prompt_text_param="", 
                       # prompt_lang_param="中文", 
                       top_k_param=5,
                       top_p_param=1.0,
                       temperature_param=1.0,
                       text_split_method_param="凑四句一切",
                       speed_factor_param=1.0,
                       seed_param=-1.0, 
                       keep_random_param=True, 
                       sample_steps_param="32",
                       ):
    """测试gradio_client连接 /inference API"""

    # 固定参考音频和文本逻辑
    fixed_ref_audio_path = os.path.abspath("server/tts_wav/1.wav")
    fixed_ref_text_path = os.path.abspath("server/tts_wav/1.txt")
    fixed_prompt_lang = "中文"

    print(f"🧪 测试Gradio Client连接: {api_url} (API: /inference)")
    print(f"📜 合成文本: '{text_to_synthesize}' ({text_lang_param})")
    # 更新打印以反映固定值
    print(f"🎤 主参考音频 (固定): {fixed_ref_audio_path}")
    # (参考文本内容会在读取后打印)
    print(f"🗣️ 参考语种 (固定): {fixed_prompt_lang}")
    print(f"🔝 Top_k: {top_k_param}")
    print(f"🎯 Top_p: {top_p_param}")
    print(f"🌡️ Temperature: {temperature_param}")
    print(f"🔪 切割方法: {text_split_method_param}")
    print(f"⏩ 语速因子: {speed_factor_param}")
    print(f"🌱 Seed: {seed_param}")
    print(f"❓ Keep Random: {keep_random_param}")
    print(f"👣 Sample Steps: {sample_steps_param}")

    # 客户端种子设置 (主要影响客户端的随机性，对服务端效果有限，但保持以防万一)
    # 服务端种子由 seed_param 控制传递给API
    if seed_param != -1.0:
        try:
            actual_client_seed = int(seed_param)
            print(f"🌱 (客户端侧)设置随机种子: {actual_client_seed}")
            random.seed(actual_client_seed)
        except ValueError:
            print(f"⚠️ 无法将seed '{seed_param}' 转为整数用于客户端random.seed().")
    else:
        print(f"🌱 (客户端侧)随机种子: 未特别设置 (使用系统随机性)")

    try:
        print(f"\n🔗 正在连接到Gradio服务... (SSL验证禁用)")
        client = Client(api_url, ssl_verify=False)
        print(f"✅ Gradio Client连接成功 (SSL验证已禁用)")

        # 检查固定参考文件路径
        if not os.path.exists(fixed_ref_audio_path):
            print(f"❌ 主参考音频文件不存在: {fixed_ref_audio_path}")
            return False
        print(f"✅ 主参考音频文件检查通过: {fixed_ref_audio_path}")

        if not os.path.exists(fixed_ref_text_path):
            print(f"❌ 参考文本文件不存在: {fixed_ref_text_path}")
            return False
        print(f"✅ 参考文本文件检查通过: {fixed_ref_text_path}")
        
        with open(fixed_ref_text_path, 'r', encoding='utf-8') as f:
            fixed_prompt_text = f.read().strip()
        print(f"📝 参考文本 (固定内容): '{fixed_prompt_text}'")

        # 构建传递给predict的参数字典
        # 该字典将包含所有API定义的参数，明确指定值或使用API文档中的默认值
        predict_params = {
            # 从函数参数或固定值获取
            "text": text_to_synthesize,
            "text_lang": text_lang_param,
            "ref_audio_path": file(fixed_ref_audio_path),
            "aux_ref_audio_paths": [], # 必需，默认为空列表
            "prompt_text": fixed_prompt_text,
            "prompt_lang": fixed_prompt_lang,
            "top_k": top_k_param, # API默认5, 可由命令行覆盖
            "top_p": top_p_param, # API默认1.0, 可由命令行覆盖
            "temperature": temperature_param, # API默认1.0, 可由命令行覆盖
            "text_split_method": text_split_method_param, # API默认"凑四句一切", 可由命令行覆盖
            "speed_factor": speed_factor_param, # API默认1.0, 可由命令行覆盖
            "seed": float(seed_param), # API默认-1.0, 可由命令行覆盖
            "keep_random": keep_random_param, # API默认True, 可由命令行覆盖
            "sample_steps": sample_steps_param, # API默认"32", 可由命令行覆盖

            # 根据API文档添加其他参数及其默认值 (这些当前不由命令行控制)
            "batch_size": 20.0,  # API Default: 20 (float)
            "ref_text_free": False, # API Default: False
            "split_bucket": True,  # API Default: True
            "fragment_interval": 0.3, # API Default: 0.3
            "parallel_infer": True, # API Default: True
            "repetition_penalty": 1.35, # API Default: 1.35
            "super_sampling": False, # API Default: False
            
            "api_name": "/inference" # 指定API端点
        }
        
        print("\n[DEBUG] 实际调用 predict 的所有参数 (包括API默认值):")
        for key, value in predict_params.items():
            if key == "ref_audio_path": 
                 print(f"   {key}: {fixed_ref_audio_path} (gradio.file object)") # 打印固定路径
            else:
                 print(f"   {key}: {value}")
        print("\n")

        print(f"🔊 开始TTS合成测试 (API: /inference)... ")
        result_tuple = client.predict(**predict_params)
        
        print(f"✅ TTS合成成功!")
        if isinstance(result_tuple, tuple) and len(result_tuple) == 2:
            output_audio_path, returned_seed = result_tuple
            print(f"📁 输出文件: {output_audio_path}")
            print(f"🌱 服务端返回的Seed: {returned_seed}")
            
            if output_audio_path and os.path.exists(output_audio_path):
                file_size = os.path.getsize(output_audio_path)
                print(f"📊 文件大小: {file_size} bytes")
                
                import shutil
                output_filename = "test_gradio_inference_output.wav"
                shutil.copy2(output_audio_path, output_filename)
                print(f"✅ 音频已复制到: {os.path.abspath(output_filename)}")
                return True
            else:
                print(f"❌ 输出文件无效或不存在: {output_audio_path}")
                return False
        else:
            print(f"❌ API返回结果格式不符合预期: {result_tuple}")
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
    # 默认参数设置区域
    api_url_cli = "http://localhost:9872"
    text_cli = "hello. im from American"
    text_lang_cli = "英文"
    # 以下三个参数不再从命令行读取，将使用函数内部的固定值
    # ref_audio_cli = os.path.abspath("server/tts_wav/1.wav") 
    # prompt_text_cli = "" 
    # prompt_lang_cli = "中文"
    top_k_cli = 5
    top_p_cli = 1.0
    temperature_cli = 1.0
    text_split_method_cli = "凑四句一切"
    speed_factor_cli = 1.0
    seed_cli = -1.0 
    keep_random_cli = True 
    sample_steps_cli = "32" 

    # 命令行参数解析 (简单版本，后续可增强)
    args = sys.argv[1:] # 跳过脚本名
    # 调整参数索引因为移除了3个参数
    if len(args) > 0: api_url_cli = args[0]
    if len(args) > 1: text_cli = args[1]
    if len(args) > 2: text_lang_cli = args[2]
    # args[3], args[4], args[5] 原用于 ref_audio, prompt_text, prompt_lang，现已移除
    if len(args) > 3: top_k_cli = int(args[3]) # 原 args[6]
    if len(args) > 4: top_p_cli = float(args[4]) # 原 args[7]
    if len(args) > 5: temperature_cli = float(args[5]) # 原 args[8]
    if len(args) > 6: text_split_method_cli = args[6] # 原 args[9]
    if len(args) > 7: speed_factor_cli = float(args[7]) # 原 args[10]
    if len(args) > 8: seed_cli = float(args[8]) # 原 args[11]
    if len(args) > 9: # 原 args[12]
        val = args[9].lower()
        if val == "true" or val == "1": keep_random_cli = True
        elif val == "false" or val == "0": keep_random_cli = False
        else: print(f"⚠️ 无效的keep_random参数值 '{args[9]}'. 使用默认值 {keep_random_cli}.")
    if len(args) > 10: sample_steps_cli = args[10] # 原 args[13]

    print("=" * 60)
    print(f"Gradio Client TTS (/inference) 测试")
    print(f"  API URL: {api_url_cli}")
    print(f"  合成文本: \"{text_cli}\" ({text_lang_cli})")
    # 更新打印以反映固定参考
    print(f"  主参考音频: (固定路径 server/tts_wav/1.wav)")
    print(f"  参考文本: (固定从 server/tts_wav/1.txt 读取)")
    print(f"  参考语种: (固定为 中文)")
    print(f"  Top_k: {top_k_cli}, Top_p: {top_p_cli}, Temperature: {temperature_cli}")
    print(f"  切割: {text_split_method_cli}, 语速: {speed_factor_cli}")
    print(f"  Seed: {seed_cli}, Keep Random: {keep_random_cli}, Sample Steps: {sample_steps_cli}")
    print("=" * 60)

    # 基本连接测试仍然有用
    # if test_simple_connection(api_url_cli): # 可以选择性运行
    #     print("\n✅ 基本连接测试通过.\n")
    
    success = test_gradio_client(
        api_url=api_url_cli,
        text_to_synthesize=text_cli,
        text_lang_param=text_lang_cli,
        # 不再传递 ref_audio_path_param, prompt_text_param, prompt_lang_param
        top_k_param=top_k_cli,
        top_p_param=top_p_cli,
        temperature_param=temperature_cli,
        text_split_method_param=text_split_method_cli,
        speed_factor_param=speed_factor_cli,
        seed_param=seed_cli,
        keep_random_param=keep_random_cli,
        sample_steps_param=sample_steps_cli
    )

    if success:
        print(f"\n🎉 Gradio Client TTS (/inference) 测试成功!")
    else:
        print(f"\n❌ Gradio Client TTS (/inference) 测试失败!")
    
    print("=" * 60) 