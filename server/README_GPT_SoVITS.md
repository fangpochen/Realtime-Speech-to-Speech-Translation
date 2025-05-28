# GPT-SoVITS 实时翻译服务器配置指南

## 概述

本项目已集成完整的GPT-SoVITS API支持，可以使用参考音频进行高质量的语音合成。

## 功能特性

- ✅ 使用参考音频和参考文本进行语音克隆
- ✅ 支持多种语言（中文、英文、日文、粤语、韩文）
- ✅ 可配置的语音合成参数
- ✅ 实时音频流传输
- ✅ 动态配置更新

## 安装依赖

```bash
pip install gradio_client
```

## 配置文件

### 参考音频设置

1. 将参考音频文件放置在 `server/tts_wav/1.wav`
2. 将参考音频的文本放置在 `server/tts_wav/1.txt`

### 配置参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `how_to_cut` | str | "凑四句一切" | 文本切分方式 |
| `top_k` | int | 15 | 采样时保留的候选词数量 |
| `top_p` | float | 1.0 | 核采样概率阈值 |
| `temperature` | float | 1.0 | 采样温度，控制随机性 |
| `speed` | float | 1.0 | 语速倍率 |
| `sample_steps` | str | "8" | 采样步数 |
| `pause_second` | float | 0.3 | 句间停顿时间 |

## 使用方法

### 基础使用

```python
from server_funasr import AudioSocketServerFunASR

# 创建服务器
server = AudioSocketServerFunASR(
    funasr_model="paraformer-zh",
    gpt_sovits_api="http://localhost:9872"
)

# 启动服务器
server.start()
```

### 配置更新

```python
# 查看当前配置
config = server.get_gpt_sovits_config()
print(config)

# 更新配置
server.update_gpt_sovits_config(
    speed=1.2,           # 语速调快20%
    temperature=0.8,     # 降低随机性
    top_k=20,           # 增加采样范围
    pause_second=0.5    # 增加句间停顿
)
```

## 支持的语言

- 中文 (`zh`)
- 英文 (`en`) 
- 日文 (`ja`)
- 粤语 (`yue`)
- 韩文 (`ko`)

## 运行示例

```bash
# 使用默认配置
python server_funasr.py

# 指定GPT-SoVITS API地址
python server_funasr.py http://localhost:9872

# 运行配置示例
python example_usage.py
```

## 注意事项

1. 确保GPT-SoVITS服务已启动并运行在指定端口
2. 参考音频建议3-10秒，音质清晰
3. 参考文本应与参考音频内容完全一致
4. 服务器监听端口4444，客户端需连接此端口

## 故障排除

### 常见问题

1. **连接失败**: 检查GPT-SoVITS服务是否正常运行
2. **音频质量差**: 调整temperature和top_k参数
3. **语速不合适**: 修改speed参数
4. **合成失败**: 检查参考音频和文本是否匹配

### 日志信息

服务器会输出详细的日志信息，包括：
- 🎤 语音识别结果
- 🔄 翻译过程
- 🔊 语音合成状态
- ✅ 音频传输确认 