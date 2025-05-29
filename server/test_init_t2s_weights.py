#!/usr/bin/env python3
"""测试gradio_client连接GPT-SoVITS API: /init_t2s_weights"""

import os
import sys
import traceback
from gradio_client import Client

def test_init_gpt_model(
    api_url="http://localhost:9872/",
    gpt_weights_path_param="GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
):
    """测试 /init_t2s_weights API"""

    print(f"🧪 测试Gradio Client连接: {api_url} (API: /init_t2s_weights)")
    print(f"🔄 欲初始化的GPT模型权重路径: {gpt_weights_path_param}")

    try:
        print(f"\n🔗 正在连接到Gradio服务... (SSL验证禁用)")
        client = Client(api_url, ssl_verify=False)
        print(f"✅ Gradio Client连接成功 (SSL验证已禁用)")

        # 构建传递给predict的参数字典
        predict_params = {
            "weights_path": gpt_weights_path_param,
            "api_name": "/init_t2s_weights"
        }
        
        print("\n[DEBUG] 实际调用 predict 的所有参数:")
        for key, value in predict_params.items():
            print(f"   {key}: {value}")
        print("\n")

        print(f"🚀 开始调用 /init_t2s_weights API... ")
        result_tuple = client.predict(**predict_params) 
        
        print(f"✅ API调用成功!")
        
        if isinstance(result_tuple, tuple):
            if len(result_tuple) >= 1: # 预期情况：至少有一个元素
                returned_value = result_tuple[0]
                print(f"📦 返回值 (元组的第一个元素): 类型: {type(returned_value)}, 值: {returned_value}")
                print("\n📊 API按预期返回了一个值。")
                return True
            elif len(result_tuple) == 0: # 收到空元组 ()
                print("ℹ️  API返回了一个空元组 ().")
                print("    这可能表示操作已在服务端执行完毕，但此API端点没有配置具体的返回值。")
                print("    为了继续测试流程，此处将空元组视作一种可能的成功指示。")
                print("    强烈建议检查服务端日志以确认模型加载是否真的成功。")
                return True
        elif result_tuple is None: # 如果 predict 直接返回 None (某些Gradio版本或情况下可能)
            print("ℹ️  API调用返回了 None.")
            print("    这可能表示操作已在服务端执行完毕，但此API端点没有配置具体的返回值。")
            print("    为了继续测试流程，此处将 None 视作一种可能的成功指示。")
            print("    强烈建议检查服务端日志以确认模型加载是否真的成功。")
            return True
        else: # 其他意外的非元组类型
            print(f"❌ API返回结果格式不符合预期 (非元组，非None): 类型: {type(result_tuple)}, 值: {result_tuple}")
            return False
            
    except Exception as e:
        print(f"❌ /init_t2s_weights API测试失败: {e}")
        print(f"🔍 详细错误信息:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 默认参数设置区域
    api_url_cli = "http://localhost:9872/"
    # 从用户提供的信息中选择一个有效的默认GPT模型路径
    gpt_weights_path_cli = "GPT_SoVITS/pretrained_models/s1v3.ckpt"

    # 命令行参数解析
    args = sys.argv[1:] # 跳过脚本名
    if len(args) > 0: api_url_cli = args[0]
    if len(args) > 1: gpt_weights_path_cli = args[1]

    print("=" * 60)
    print(f"Gradio Client API Test: /init_t2s_weights")
    print(f"  API URL: {api_url_cli}")
    print(f"  GPT Weights Path: {gpt_weights_path_cli}")
    print("=" * 60)
    
    success = test_init_gpt_model(
        api_url=api_url_cli,
        gpt_weights_path_param=gpt_weights_path_cli
    )

    if success:
        print(f"\n🎉 /init_t2s_weights API 测试在客户端层面被视为成功 (请务必检查服务端日志确认模型加载状态)!")
    else:
        print(f"\n❌ /init_t2s_weights API 测试失败!")
    
    print("=" * 60) 