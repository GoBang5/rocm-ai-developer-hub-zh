> 原文作者：Victoria Godsoe  
> [AMD 官方教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/voice_pipeline_rag_ollama.html) · [固定版本源码](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/inference/voice_pipeline_rag_ollama.ipynb)  
> 上游版本：`0071bfd4bd3bd9525405c09a6ae69c983b09312a` · 同步日期：2026-10-12  
> 翻译状态：机器初译，待逐篇人工校对；代码与命令保持原文，未在 AMD GPU 上运行验证。

<!-- cell:1 -->

<a id="building-ai-pipelines-for-voice-assistants-using-rocm-llamaindex-and-rag"></a>

# 为语音助手构建 AI 流水线
**作者**：Victoria Godsoe
**知识水平**：初学者

以下Notebook演示了如何将 AMD GPU 与 [LlamaIndex](https://docs.llamaindex.ai/en/stable/) 和 检索增强生成 (RAG) 结合使用。它获取输入录音，将其转录为文本，将转录的文本发送到 RAG 模型，并生成文本格式的响应，然后将其转换为语音并保存为音频文件。

<a id="prerequisites"></a>

## 前置条件

本教程是使用以下设置进行开发和测试的。

<a id="operating-system"></a>

### 操作系统

* **Ubuntu 22.04**：确保您的系统运行的是 Ubuntu 版本 22.04。

<a id="hardware"></a>

### 硬件

* **AMD GPU**：本教程在 AMD Instinct™ MI300X 和 AMD Radeon™ W7900 上进行了测试。确保您使用的是支持 ROCm 的 AMD GPU，并且您的系统满足[官方要求](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)。

<a id="software"></a>

### 软件

本教程使用以下设置在 AMD Radeon 和 AMD Instinct GPU 上进行了测试：
- ROCm 6.2.0
-Python 3.10
- PyTorch 2.3.0

<a id="objectives"></a>

### 目标

完成本教程后，您应该了解以下概念：

* 多模型管道
* AMD GPU 上具有 ROCm 的 LlamaIndex

<a id="prepare-the-inference-environment"></a>

## 准备推理环境

要设置 推理 环境，请执行以下步骤：

<details markdown="1">
<summary>查看英文原文 · 单元 1</summary>

# Building AI pipelines for voice assistants using ROCm, LlamaIndex, and RAG

**Author**: Victoria Godsoe  
**Knowledge level**: Beginner

The following notebook demonstrates how to use AMD GPUs with [LlamaIndex](https://docs.llamaindex.ai/en/stable/) and Retrieval-Augmented Generation (RAG). It takes an input audio recording, transcribes it to text, sends the transcribed text to the RAG model, and generates a response in text format, which is then converted to speech and saved as an audio file.

## Prerequisites

This tutorial was developed and tested using the following setup.

### Operating system

* **Ubuntu 22.04**: Ensure your system is running Ubuntu version 22.04.

### Hardware

* **AMD GPUs**: This tutorial was tested on an AMD Instinct™ MI300X and an AMD Radeon™ W7900. Ensure you are using an AMD GPU with ROCm support and that your system meets the [official requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

### Software

This tutorial was tested on both AMD Radeon and AMD Instinct GPUs using the following setup:
- ROCm 6.2.0
- Python 3.10
- PyTorch 2.3.0

### Objectives

After completing this tutorial, you should understand the following concepts:

* Multi-model pipeline
* LlamaIndex with ROCm on AMD GPUs

## Prepare the inference environment

To set up the inference environment, follow these steps:

</details>

<!-- cell:2 -->

1、创建conda环境：
   ```bash
   conda create -n rocm python=3.10
   ```

1.激活环境：
   ```bash
   conda activate rocm
   ```

1. 安装 PyTorch for ROCm 软件：
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
   ```

1. 安装 Ollama（如果之前未安装）。此步骤需要 `curl`：
   ```bash
   sudo apt install curl -y
   curl -fsSL https://ollama.com/install.sh | sh
   ```

1. 启动 Ollama 服务器（如果尚未运行）：
   ```bash
   ollama serve &
   ```

1.用Ollama拉动`llama3`：
   ```bash
   ollama pull llama3
   ```

1. 安装示例依赖项：
   ```bash
   pip install llama-index llama-index-llms-ollama llama-index-embeddings-ollama llama-index-embeddings-huggingface openai-whisper transformers ChatTTS
   ```

1. 包含音频文件（例如，`summarize_question.wav`）。将其放在当前工作目录中。

<details markdown="1">
<summary>查看英文原文 · 单元 2</summary>

1. Create a conda environment:
   ```bash
   conda create -n rocm python=3.10
   ```

1. Activate the environment:
   ```bash
   conda activate rocm
   ```

1. Install the PyTorch for ROCm software:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
   ```

1. Install Ollama (if not previously installed). This step requires `curl`:
   ```bash
   sudo apt install curl -y
   curl -fsSL https://ollama.com/install.sh | sh
   ```

1. Launch the Ollama server if it isn't already running:
   ```bash
   ollama serve &
   ```

1. Pull `llama3` with Ollama:
   ```bash
   ollama pull llama3
   ```

1. Install the example dependencies:
   ```bash
   pip install llama-index llama-index-llms-ollama llama-index-embeddings-ollama llama-index-embeddings-huggingface openai-whisper transformers ChatTTS
   ```

1. Include an audio file (for example, `summarize_question.wav`). Place it in the current working directory.

</details>

<!-- cell:3 -->

<a id="install-and-launch-jupyter"></a>

### 安装并启动 Jupyter

在 Docker 容器内，使用以下命令安装 Jupyter：

```bash
pip install jupyter
```

然后启动Jupyter服务器：

```bash
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**注意**：在运行上述命令之前，请确保您的系统上尚未使用端口 `8888`。如果是，您可以通过将 `--port=8888` 替换为其他端口号来指定其他端口，例如 `--port=8890`。

<details markdown="1">
<summary>查看英文原文 · 单元 3</summary>

### Install and launch Jupyter

Inside the Docker container, install Jupyter using the following command:

```bash
pip install jupyter
```

Then start the Jupyter server:

```bash
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**Note**: Ensure port `8888` is not already in use on your system before running the above command. If it is, you can specify a different port by replacing `--port=8888` with another port number, for example, `--port=8890`.

</details>

<!-- cell:4 -->

<a id="import-the-packages"></a>

## 导入包

导入以下包：

* **os**：依赖于操作系统的功能。
* **whisper**：语音识别库。
* **torch**：用于张量计算和深度学习的 PyTorch 库。
* **llama_index.core**：Llama 索引的核心功能。
* **llama_index.embeddings.huggingface**：支持嵌入 HuggingFace。
* **llama_index.llms.ollama**：Ollama 语言模型的功能。
* **ChatTTS**：文本到语音转换库。
* **torchaudio**：音频处理库。
* **IPython.display**：用于在 Jupyter Notebook中显示音频。

<details markdown="1">
<summary>查看英文原文 · 单元 4</summary>

## Import the packages

Import the following packages:

* **os**: Operating-system-dependent functionality.
* **whisper**: A speech-recognition library.
* **torch**: A PyTorch library for tensor computations and deep learning.
* **llama_index.core**: Core functionality for the Llama Index.
* **llama_index.embeddings.huggingface**: Support for embedding HuggingFace.
* **llama_index.llms.ollama**: Functionality for the Ollama language model.
* **ChatTTS**: A text-to-speech conversion library.
* **torchaudio**: An audio-processing library.
* **IPython.display**: For displaying audio in Jupyter notebooks.

</details>

```python
# System Imports
import os
import numpy as np
import re

# Imports for Speech to Text
import whisper
import torch

# Imports for RAG Model
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# Imports for Text to Speech
import ChatTTS
import torchaudio
from IPython.display import Audio
```

<!-- cell:6 -->

<a id="set-up-the-environment"></a>

## 设置环境

1. （可选）设置环境变量以启用 PyTorch ROCm 中的实验功能。
2. 验证 PyTorch 版本和 GPU 可用性。
3. 选择计算设备：
   - 使用 GPU（如果可用）并打印其属性。
   - 否则回退到 CPU。

<details markdown="1">
<summary>查看英文原文 · 单元 6</summary>

## Set up the environment

1. Optionally, set the environment variables to enable experimental features in PyTorch ROCm.
2. Verify the PyTorch version and GPU availability.
3. Select the computation device:
   - Use the GPU if available and print its properties.
   - Fall back to the CPU otherwise.

</details>

```python
# Set the environment variable for experimental features (optional)
os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '1'
os.environ['HIP_VISIBLE_DEVICES'] = "0"

print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Check GPU availability and properties
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU (no GPU detected)")
```

<!-- cell:8 -->

<a id="transcribe-speech-to-text"></a>

## 将语音转录为文本

以下部分使用 Whisper 模型执行语音到文本的转录。

首先，下载本教程的示例音频文件。

<details markdown="1">
<summary>查看英文原文 · 单元 8</summary>

## Transcribe speech to text

The following section performs speech-to-text transcription using the Whisper model.

First, download the sample audio file for this tutorial.

</details>

```python
!curl -L https://raw.githubusercontent.com/ROCm/gpuaidev/main/docs/notebooks/assets/summarize_question.wav -o summarize_question.wav
```

<!-- cell:10 -->

加载音频文件并将语音内容转录为文本。

<details markdown="1">
<summary>查看英文原文 · 单元 10</summary>

Load the audio file and transcribe the speech content into text.

</details>

```python
AUDIO_FILE = "summarize_question.wav"
Audio(AUDIO_FILE, rate=24_000, autoplay=True)
```

<!-- cell:12 -->

现在将演讲内容转录为文本。

<details markdown="1">
<summary>查看英文原文 · 单元 12</summary>

Now transcribe the speech content into text.

</details>

```python
# Install ffmpeg (required by openai-whisper for audio decoding)
!apt-get update -qq && apt-get install -y -qq ffmpeg

# Speech-to-Text with Whisper
try:
    model = whisper.load_model("base")
    result = model.transcribe(AUDIO_FILE)
    input_text = result["text"]
    print(f"Transcribed text: {input_text}")
except Exception as e:
    print(f"Error in speech-to-text: {e}")
    exit(1)
```

<!-- cell:14 -->

<a id="add-the-rag-model"></a>

## 添加 RAG 模型

要使用 RAG 模型，请提供您希望 LLM 用于查询的上下文。此示例配置为使用 `data` 文件夹中的文档。如果您还没有任何文档，您可以添加自己的文档或下载以下文档。

<details markdown="1">
<summary>查看英文原文 · 单元 14</summary>

## Add the RAG model

To use a RAG model, provide the context that you'd like the LLM to use for the queries. This example is configured to use the documents in the `data` folder. If you don't have any documents yet, you can add your own or download the one below.

</details>

```python
DATA_DIR = "./data"

# Check if the data directory exists, and create it if it doesn't
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"Data directory '{DATA_DIR}' created. Please add a file of your choosing or use the cell below to download sample text.")
    exit(1)
else:
    # Check if data directory is empty
    if not os.listdir(DATA_DIR):
        print(f"Data directory '{DATA_DIR}' is empty. Please add a file of your choosing or use the cell below to download sample text.")
        exit(1)
```

<!-- cell:16 -->

如果 `data` 目录为空，请运行以下单元格：

<details markdown="1">
<summary>查看英文原文 · 单元 16</summary>

If the `data` directory is empty, run the following cell:

</details>

```python
## OPTIONAL - Run this cell if your data directory is empty
!mkdir -p data && curl -L https://www.gutenberg.org/cache/epub/11/pg11.txt -o data/pg11.txt
```

<!-- cell:18 -->

验证数据文件现在存在于 `data` 目录中。

<details markdown="1">
<summary>查看英文原文 · 单元 18</summary>

Verify the data file now exists in the `data` directory.

</details>

```python
# View the files in your data directory
print("Files in data directory:", os.listdir("data"))
documents = SimpleDirectoryReader(DATA_DIR).load_data()
```

<!-- cell:20 -->

对于嵌入模型，请使用 `HuggingFaceEmbedding` 中的“bge-base”。确认 Ollama 服务器正在运行，因为它为 LLM 提供 Llama-3。  

接下来，从加载的文档创建 `VectorStoreIndex` 并使用索引初始化查询引擎。然后使用 Whisper 模型的文本输出发出查询。打印响应，以便您可以将其与下一步中的音频输出进行比较。

<details markdown="1">
<summary>查看英文原文 · 单元 20</summary>

For the embedding model, use "bge-base" from `HuggingFaceEmbedding`. Confirm that the Ollama server is running because it supplies Llama-3 for the LLM.  

Next, create a `VectorStoreIndex` from the loaded documents and initialize a query engine with the index. Then issue your query using the text output from the Whisper model. Print the response so you can compare it against the audio output in the next step.

</details>

```python
# Initialize embedding and LLM models
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

try:
    Settings.llm = Ollama(model="llama3", request_timeout=360.0)
except Exception as e:
    print(f"Error connecting to Ollama server: {e}")
    exit(1)

# Build and query the vector index
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(streaming=True, response_mode="compact", similarity_top_k=3)
response = query_engine.query(input_text)

# Function to convert StreamingResponse to string
def streaming_response_to_string(streaming_response):
    text = ""
    for chunk in streaming_response.response_gen:
        if isinstance(chunk, dict) and "text" in chunk:
            text += chunk["text"]
        else:
            text += str(chunk)
    return text

# Convert response to string
response_text = streaming_response_to_string(response)
print(f"Generated response: {response_text}")
```

<!-- cell:22 -->

<a id="perform-text-to-speech-conversion"></a>

## 执行文本到语音的转换

以下示例使用 ChatTTS 库执行文本到语音转换，并将输出音频保存到文件中。

此示例使用以下常量：
* `OUTPUT_AUDIO_FILE` (`str`): 输出音频文件的名称。
* `SAMPLE_RATE` (`int`)：输出音频文件的采样率。

它提供以下功能：
* 初始化一个`ChatTTS.Chat`对象。
* 无需编译即可加载聊天模型，以加快加载速度。 （设置 `compile=True` 以获得更好的性能。）
* 将上一步的响应文本转换为语音。
* 使用torchaudio将生成的音频保存到指定的输出文件。

<details markdown="1">
<summary>查看英文原文 · 单元 22</summary>

## Perform text-to-speech conversion

The following example performs text-to-speech conversion using the ChatTTS library and saves the output audio to a file.

This example uses the following constants:
* `OUTPUT_AUDIO_FILE` (`str`): The name of the output audio file.
* `SAMPLE_RATE` (`int`): The sample rate for the output audio file.

It provides the following functionality:
* Initializes a `ChatTTS.Chat` object.
* Loads the chat model without compilation for faster loading. (Set `compile=True` for better performance.)
* Converts the response text from the previous step to speech.
* Saves the generated audio to the specified output file using torchaudio.

</details>

```python
OUTPUT_AUDIO_FILE = "voice_pipeline_response.wav"
SAMPLE_RATE = 24000

# Text cleanup function for TTS
def sanitize_input(text):
    sanitized_text = text.replace('-', '')  # Remove hyphens
    sanitized_text = sanitized_text.replace('(', '').replace(')', '')  # Remove parentheses
    return sanitized_text.strip()

# Text-to-Speech processing
try:
    sanitized_response = re.sub(r"[^a-zA-Z0-9.,?! ]", "", response_text)  # Remove special characters
    print(f"Sanitized response for TTS: {sanitized_response}")
    sanitized_response = [sanitized_response]

    chat = ChatTTS.Chat()
    chat.load(compile=False) # Set to True for better performance

    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb = chat.sample_random_speaker(),
    )

    wavs = chat.infer(
        sanitized_response,
        params_infer_code=params_infer_code,
    )
    try:
        torchaudio.save(OUTPUT_AUDIO_FILE, torch.from_numpy(wavs[0]).unsqueeze(0), SAMPLE_RATE)
    except:
        torchaudio.save(OUTPUT_AUDIO_FILE, torch.from_numpy(wavs[0]), SAMPLE_RATE)

except Exception as e:
    print(f"Error in text-to-speech: {e}")
    exit(1)

finally:
    if 'chat' in locals():
        chat.unload()
```

<!-- cell:24 -->

播放以下单元格以收听生成的语音。

<details markdown="1">
<summary>查看英文原文 · 单元 24</summary>

Play the following cell to hear the generated speech.

</details>

```python
Audio(wavs[0], rate=24_000, autoplay=True)
```
