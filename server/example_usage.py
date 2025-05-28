#!/usr/bin/env python3
"""
GPT-SoVITS实时翻译服务器使用示例
"""

from server_funasr import AudioSocketServerFunASR
import time

def main():
    # 创建服务器实例
    server = AudioSocketServerFunASR(
        funasr_model="paraformer-zh",
        gpt_sovits_api="http://localhost:9872"
    )
    
    # 查看当前配置
    print("📋 当前GPT-SoVITS配置:")
    config = server.get_gpt_sovits_config()
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 更新配置示例
    print("\n🔧 更新配置示例:")
    server.update_gpt_sovits_config(
        speed=1.2,           # 语速调快
        temperature=0.8,     # 降低随机性
        top_k=20,           # 增加采样范围
        pause_second=0.5    # 增加句间停顿
    )
    
    # 启动服务器
    print("\n🚀 启动服务器...")
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n⏹️  服务器已停止")

if __name__ == "__main__":
    main() 