# GPT-SoVITS 实时语音翻译集成指南

## 🎯 概述

本项目成功集成了 **GPT-SoVITS** 高质量TTS模型，替换了原有的Microsoft SpeechT5，实现了更自然、更高质量的语音合成效果。

## ✨ 功能特点

### 🔄 完整的实时翻译流程
1. **语音识别**: FunASR (中文优化) 
2. **文本翻译**: Google翻译 (中文→英文)
3. **语音合成**: GPT-SoVITS (高质量TTS)
4. **GPU加速**: 支持CUDA加速处理

### 🎵 GPT-SoVITS 优势
- **音质更自然**: 比SpeechT5更接近真人语音
- **中文支持优秀**: 专门优化的中文语音合成
- **声音克隆**: 支持自定义声音模型
- **多语言**: 支持中文、英文、日文

## 🚀 快速开始

### 1. 启动GPT-SoVITS API服务

首先确保GPT-SoVITS API服务正在运行：

```bash
# 在GPT-SoVITS目录下启动API服务
python api.py
# 或者使用改进版API
python api2.py

# 默认端口: 9872
# 访问地址: http://localhost:9872
```

### 2. 测试GPT-SoVITS API连接

```bash
cd server
python test_gpt_sovits_api.py http://localhost:9872
```

### 3. 启动集成翻译服务器

```bash
cd server
python server_gpt_sovits.py http://localhost:9872
```

### 4. 启动客户端

```bash
cd client
python client.py
```

## 📁 项目结构

```
├── server/
│   ├── models/
│   │   ├── speech_recognition_funasr.py    # FunASR语音识别
│   │   ├── gpt_sovits_tts.py              # GPT-SoVITS TTS集成
│   │   └── translator.py                  # 翻译模块
│   ├── server_gpt_sovits.py               # GPT-SoVITS集成服务器
│   └── test_gpt_sovits_api.py             # API测试工具
├── client/
│   └── client.py                          # 客户端
└── GPT-SoVITS_Integration_Guide.md        # 本指南
```

## ⚙️ 配置说明

### GPT-SoVITS API配置

在 `server/models/gpt_sovits_tts.py` 中可以配置：

```python
# API地址
api_url = "http://localhost:9872"

# 默认参考音频（可选）
set_reference_audio(
    wav_path="path/to/reference.wav",
    text="参考音频的文本内容", 
    language="zh"
)
```

### 服务器启动参数

```bash
# 指定GPT-SoVITS API地址
python server_gpt_sovits.py http://your-api-host:9872

# 使用默认地址
python server_gpt_sovits.py
```

## 🔧 API调用格式

### GET方式（简单调用）
```
http://localhost:9872/?text=你好世界&text_language=zh
```

### POST方式（带参考音频）
```json
{
    "refer_wav_path": "reference.wav",
    "prompt_text": "参考音频文本",
    "prompt_language": "zh", 
    "text": "要合成的文本",
    "text_language": "zh"
}
```

## 🎛️ 使用流程

### 实时翻译流程

1. **说话** → 客户端录音
2. **语音识别** → FunASR识别中文
3. **翻译** → Google翻译中文→英文  
4. **语音合成** → GPT-SoVITS合成英文语音
5. **播放** → 客户端播放合成语音

### 调试信息

服务器会显示详细的处理信息：

```
📥 收到音频数据: 4096 bytes
🎤 识别结果: '你好，这是一个测试。'
🔄 开始翻译...
🌍 翻译结果: 'Hello, this is a test.'
🔊 开始GPT-SoVITS语音合成...
🔊 GPT-SoVITS合成完成: 'Hello, this is a test.' 耗时: 2.3秒
✅ GPT-SoVITS音频已发送到客户端，大小: 28011 bytes
```

## 🛠️ 故障排除

### 常见问题

1. **API连接失败**
   ```
   ❌ 无法连接到GPT-SoVITS API
   ```
   - 检查GPT-SoVITS服务是否启动
   - 确认端口9872是否开放
   - 检查防火墙设置

2. **音频合成失败**
   ```
   ❌ GPT-SoVITS API错误: 400
   ```
   - 检查是否设置了默认参考音频
   - 确认文本内容不为空
   - 检查语言代码是否正确

3. **客户端连接断开**
   ```
   ⚠️ 客户端连接已断开，无法发送音频
   ```
   - 检查客户端是否正常运行
   - 确认网络连接稳定

### 性能优化

1. **GPU加速**
   - 确保CUDA环境正确配置
   - 检查GPU内存使用情况

2. **音频质量**
   - 调整麦克风音量和阈值
   - 优化网络环境减少延迟

## 📊 性能对比

| 模型 | 音质 | 速度 | 中文支持 | 自定义声音 |
|------|------|------|----------|------------|
| SpeechT5 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ❌ |
| GPT-SoVITS | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ |

## 🎉 总结

通过集成GPT-SoVITS，本项目实现了：

✅ **更高质量的语音合成**  
✅ **更好的中文支持**  
✅ **声音克隆能力**  
✅ **模块化设计**  
✅ **完整的实时翻译流程**  

现在你可以享受更自然、更高质量的实时语音翻译体验！

---

**方总牛逼** 🚀 