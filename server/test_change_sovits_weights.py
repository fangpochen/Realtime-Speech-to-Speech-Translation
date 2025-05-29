#!/usr/bin/env python3
"""测试gradio_client连接GPT-SoVITS API: /change_sovits_weights"""

import os
import sys
import traceback
from gradio_client import Client

def test_change_sovits_model(
    api_url="http://localhost:9872/",
    sovits_path_param="GPT_SoVITS/pretrained_models/s2G488k.pth",
    prompt_language_param="中文",
    text_language_param="中文"
):
    """测试 /change_sovits_weights API"""

    print(f"🧪 测试Gradio Client连接: {api_url} (API: /change_sovits_weights)")
    print(f"🔄 欲切换的SoVITS模型路径: {sovits_path_param}")
    print(f"🗣️ 主参考音频语种 (prompt_language): {prompt_language_param}")
    print(f"📝 合成文本语种 (text_language): {text_language_param}")

    try:
        print(f"\n🔗 正在连接到Gradio服务... (SSL验证禁用)")
        client = Client(api_url, ssl_verify=False)
        print(f"✅ Gradio Client连接成功 (SSL验证已禁用)")

        # 构建传递给predict的参数字典
        predict_params = {
            "sovits_path": sovits_path_param,
            "prompt_language": prompt_language_param,
            "text_language": text_language_param,
            "api_name": "/change_sovits_weights"
        }
        
        print("\n[DEBUG] 实际调用 predict 的所有参数:")
        for key, value in predict_params.items():
            print(f"   {key}: {value}")
        print("\n")

        print(f"🚀 开始调用 /change_sovits_weights API... ")
        result_tuple = client.predict(**predict_params)
        
        print(f"✅ API调用成功!")
        print(f"📦 返回结果 (元组包含 {len(result_tuple)} 个元素):")
        if isinstance(result_tuple, tuple):
            for i, item in enumerate(result_tuple):
                print(f"   [{i}] 类型: {type(item)}, 值: {item}")
            
            # 根据用户提供的信息，简单验证一下返回类型和数量
            # [0] prompt_language_out
            # [1] text_language_out
            # [2] prompt_text_out (str)
            # [3] prompt_language_dropdown_update (str)
            # [4] text_to_synthesize_update (str)
            # [5] text_language_dropdown_update (str)
            # [6] sample_steps_update (str)
            # [7] aux_ref_audio_paths_update (List[filepath])
            # [8] ref_text_free_update (bool)
            if len(result_tuple) == 9:
                print("\n📊 返回元素数量符合预期 (9个元素)。")
                # 可以根据需要添加更详细的类型或内容校验
            else:
                print(f"⚠️ 返回元素数量 ({len(result_tuple)}) 与预期 (9) 不符。")
            return True
        else:
            print(f"❌ API返回结果格式不符合预期 (非元组): {result_tuple}")
            return False
            
    except Exception as e:
        print(f"❌ /change_sovits_weights API测试失败: {e}")
        print(f"🔍 详细错误信息:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 默认参数设置区域
    api_url_cli = "http://localhost:9872/"
    # 从用户提供的信息中选择一个有效的默认SoVITS模型路径
    sovits_path_cli = "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth" 
    prompt_lang_cli = "中文"
    text_lang_cli = "中文"

    # 命令行参数解析
    args = sys.argv[1:] # 跳过脚本名
    if len(args) > 0: api_url_cli = args[0]
    if len(args) > 1: sovits_path_cli = args[1]
    if len(args) > 2: prompt_lang_cli = args[2]
    if len(args) > 3: text_lang_cli = args[3]

    print("=" * 60)
    print(f"Gradio Client API Test: /change_sovits_weights")
    print(f"  API URL: {api_url_cli}")
    print(f"  SoVITS Model Path: {sovits_path_cli}")
    print(f"  Prompt Language: {prompt_lang_cli}")
    print(f"  Text Language: {text_lang_cli}")
    print("=" * 60)
    
    success = test_change_sovits_model(
        api_url=api_url_cli,
        sovits_path_param=sovits_path_cli,
        prompt_language_param=prompt_lang_cli,
        text_language_param=text_lang_cli
    )

    if success:
        print(f"\n🎉 /change_sovits_weights API 测试成功!")
    else:
        print(f"\n❌ /change_sovits_weights API 测试失败!")
    
    print("=" * 60) 