> 原文作者：Charles Yang  
> [AMD 官方教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/mxfp4_quantization_quark_vllm.html) · [固定版本源码](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/gpu_dev_optimize/mxfp4_quantization_quark_vllm.ipynb)  
> 上游版本：`0071bfd4bd3bd9525405c09a6ae69c983b09312a` · 同步日期：2026-10-12  
> 翻译状态：机器初译，待逐篇人工校对；代码与命令保持原文，未在 AMD GPU 上运行验证。

[中文 Notebook](../../notebooks/gpu_dev_optimize/mxfp4_quantization_quark_vllm.ipynb) · [英文原文](../../original/gpu_dev_optimize/mxfp4_quantization_quark_vllm.md) · [中英对照](../../bilingual/gpu_dev_optimize/mxfp4_quantization_quark_vllm.md)

---

<a id="accelerating-llama33-70b-with-quark-mxfp4-quantization-for-vllm"></a>

# 使用 Quark MXFP4 量化运行 vLLM
**作者**：Charles Yang
**知识水平**：初学者

本教程介绍如何对 量化 使用 MXFP4（微缩放浮点 4）数据类型。

AMD [Quark](https://quark.docs.amd.com/latest/) 是一个灵活而强大的 量化 工具包，可以生成在 AMD GPU 上运行的高性能量化模型。 Quark 专门支持量化具有权重、激活和 kv 缓存 量化 的大型语言模型，以及 AWQ、GPTQ、Rotation 和 SmoothQuant 等尖端 量化 算法。

MXFP4 是一种低位表示格式，用于通过在值块之间共享缩放因子来压缩神经网络中的权重或激活。
具体来说，对于每个包含 32 个值的块（例如，`float32` 权重），
每个值都使用四位表示。它通常采用以下格式进行编码：

* 1位：符号
* 2位：指数
* 1 位：尾数

![mxfp4数据格式](../../original/assets/mxfp4-image1.png)

每块 32 个值存储一个共享的 8 位比例因子。
比例尺是块级共享的二次幂因子，用于近似原始浮点值。

本教程将指导您设置 Quark、将 LLM 模型量化为 MXFP4，以及使用 [ROCm](https://rocm.docs.amd.com/en/latest/index.html) 软件堆栈在 AMD Instinct™ GPU 上运行 MXFP4 模型。了解如何配置 Quark 参数以实现不同的模型精度，并使用不同的 量化 算法验证性能。

<a id="prerequisites"></a>

## 前置条件

本教程是使用以下设置进行开发和测试的。 

<a id="operating-system"></a>

### 操作系统

* **Ubuntu 24.04**：确保您的系统运行的是 Ubuntu 版本 24.04。

<a id="hardware"></a>

### 硬件

* **AMD Instinct™ MI355 GPU**：本教程需要 AMD Instinct MI355X GPU，它为 MXFP4 量化 格式提供本机支持，并确保最佳兼容性和性能。

<a id="software"></a>

### 软件

* **ROCm 7.0**：按照 [ROCm 安装指南](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html) 安装并验证 ROCm。安装后，使用以下命令确认您的设置：

    ``` bash
    amd-smi
    ```

    此命令列出了您的 AMD GPU 以及相关详细信息。
    
    **注意**：对于 ROCm 6.4 及更早版本，请改用 `rocm-smi` 命令。

* **Docker**：确保 Docker 已正确安装和配置。请遵循适用于您的操作系统的 Docker 安装指南。

   **注意**：确保正确配置 Docker 权限。要配置权限以允许非 root 访问，请运行以下命令：

   ``` bash
   sudo usermod -aG docker $USER
   newgrp docker
   ```

   验证 Docker 是否正常工作：

   ``` bash
   docker run hello-world
   ```

<a id="hugging-face-api-access"></a>

### Hugging Face API 访问

* 从[Hugging Face](https://huggingface.co)获取API令牌用于下载模型。
* 确保 Hugging Face API 令牌具有访问 [Meta Llama 检查点](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) 所需的权限和批准。

<a id="env-setup"></a>

<a id="environment-setup-with-docker-and-rocm"></a>

## 使用 Docker 和 ROCm 设置环境

按照以下步骤设置环境、启动 Jupyter Notebooks 并安装依赖项。

<a id="1-launch-the-docker-container"></a>

### 1.启动Docker容器

启动 Docker 容器。从您的主机运行以下命令：

``` bash
docker run -it --rm \
  --privileged \
  --network=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size 8G \
  -v $(pwd):/workspace \
  -w /workspace/notebooks \
  vllm/vllm-openai-rocm:v0.18.0 bash
```

**注意**：该命令将当前目录挂载到容器中的`/workspace`目录中。确保在运行 Docker 命令之前将Notebook文件复制到此目录，或者在启动后上传到 Jupyter Notebook 环境。保存终端输出中提供的令牌或 URL，以便从 Web 浏览器访问Notebook。您可以从 [AI Developer Hub GitHub 存储库](https://github.com/ROCm/gpuaidev) 下载此Notebook。

<a id="launch-jupyter"></a>

<a id="2-launch-jupyter-notebooks-in-the-container"></a>

### 2. 在容器中启动 Jupyter Notebooks

在 Docker 容器内，使用以下命令安装 Jupyter：

``` bash
pip install jupyter
```

启动 Jupyter 服务器：

``` bash
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**注意**：在运行上述命令之前，请确保您的系统上尚未使用端口 `8888`。如果是，您可以通过将 `--port=8888` 替换为其他端口号来指定其他端口，例如 `--port=8890`。

<a id="install-deps"></a>

<a id="3-installing-dependencies"></a>

### 3.安装依赖项

接下来，安装 CMake 和 Quark。选择 PyTorch 的 CPU 轮，以便 Quark 可以在没有 GPU 的Notebook上运行。这种方法速度较慢，但​​适合尝试 Quark。从 PyPI 安装 Quark，它会引入所需的依赖项。

在 Docker 容器中运行的 Jupyter Notebook 中运行以下命令：

```python
%%bash
# Install basics
apt-get update
apt-get install -y unzip wget
pip install cmake jupyter ipython ipywidgets
pip install 'huggingface-hub<1.0' evaluate accelerate datasets pillow transformers zstandard lm-eval

# Download and unzip AMD Quark examples (config files and example scripts)
wget -O amd_quark-0.11.1.zip https://download.amd.com/opendownload/Quark/amd_quark-0.11.1.zip
unzip -o amd_quark-0.11.1.zip

# Install AMD Quark Python package from PyPI 
pip install amd-quark==0.11.1
```

<a id="4-provide-your-hugging-face-token"></a>

### 4. 提供您的拥抱令牌

您需要 Hugging Face API 令牌才能访问 Llama-3.3-70B。在 [Hugging Face Tokens](https://huggingface.co/settings/tokens) 生成您的令牌并请求访问 [Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)。令牌通常以“hf_”开头。

在 Jupyter Notebook中运行以下交互式块来设置令牌：

```python
from huggingface_hub import notebook_login, HfApi

# Prompt the user to log in
notebook_login()
```

验证您的令牌是否被正确接受：

```python
# Validate the token
try:
    api = HfApi()
    user_info = api.whoami()
    print(f"Token validated successfully! Logged in as: {user_info['name']}")
except Exception as e:
    print(f"Token validation failed. Error: {e}")
```

<a id="quantization-process"></a>

## 量化 流程

安装Quark后，请按照此示例学习如何使用Quark。
Quark 量化 流程由以下五个步骤组成，解释如下：

1. 加载模型。
2. 准备校准数据加载器。
3. 设置量化 配置。
4. 量化并导出模型。
5. 在 vLLM 中进行评估。

<a id="1-load-the-model"></a>

### 1.加载模型
Quark 使用 Transformer 来获取模型和分词器。

```python
### Loading Model
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
MAX_SEQ_LEN = 512
GROUP_SIZE=32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto",
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, model_max_length=MAX_SEQ_LEN)
tokenizer.pad_token = tokenizer.eos_token
```

<a id="2-prepare-the-calibration-dataloader"></a>

### 2. 准备校准数据加载器
Quark 使用 PyTorch DataLoader 加载校准数据。更多如何高效使用标定数据集的详细信息，请参见[添加标定数据集](https://quark.docs.amd.com/latest/pytorch/calibration_datasets.html)。

```python
### Dataset
from datasets import load_dataset
from torch.utils.data import DataLoader

BATCH_SIZE = 1
NUM_CALIBRATION_DATA = 512

# Load the dataset and get calibration data.
dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
text_data = dataset["text"][:NUM_CALIBRATION_DATA]

tokenized_outputs = tokenizer(text_data, return_tensors="pt",
    padding=True, truncation=True, max_length=MAX_SEQ_LEN)
calib_dataloader = DataLoader(tokenized_outputs['input_ids'],
    batch_size=BATCH_SIZE, drop_last=True)
```

<a id="3-set-the-quantization-configuration"></a>

### 3.设置量化配置

下一步是设置 量化 配置。更多详情请参见[Quark配置指南](https://quark.docs.amd.com/latest/pytorch/basic_usage_pytorch.html)。此示例在权重、激活和 kv 缓存上使用 `FP8` 每张量 量化，而 量化 算法是 AutoSmoothQuant。

```python
### Quant Configuration
def FP4_PER_GROUP_SYM_SPEC(group_size, scale_format="e8m0", scale_calculation_mode="even", is_dynamic=True):
    return FP4PerGroupSpec(ch_axis=-1,
                           group_size=group_size,
                           scale_format=scale_format,
                           scale_calculation_mode=scale_calculation_mode,
                           is_dynamic=is_dynamic).to_quantization_spec()

from quark.torch.quantization import (Config, QuantizationConfig,
                                     FP4PerGroupSpec,
                                     OCP_MXFP4Spec,
                                     FP8E4M3PerTensorSpec,
                                     load_quant_algo_config_from_file)

# Define kv-cache fp8/per-tensor/static spec.
FP8_PER_TENSOR_SPEC = FP8E4M3PerTensorSpec(observer_method="min_max",
    is_dynamic=False).to_quantization_spec()

# Define global quantization config, input tensors and weight apply FP4_PER_GROUP_SYM_SPEC.
global_quant_config = QuantizationConfig(input_tensors=FP4_PER_GROUP_SYM_SPEC(GROUP_SIZE, "e8m0", "even", True), \
        weight=FP4_PER_GROUP_SYM_SPEC(GROUP_SIZE, "e8m0", "even", False))

# Define quantization config for kv-cache layers, output tensors apply FP8_PER_TENSOR_SPEC.
KV_CACHE_SPEC = FP8_PER_TENSOR_SPEC
kv_cache_layer_names_for_llama = ["*k_proj", "*v_proj"]
kv_cache_quant_config = {name :
    QuantizationConfig(input_tensors=global_quant_config.input_tensors,
                       weight=global_quant_config.weight,
                       output_tensors=KV_CACHE_SPEC)
    for name in kv_cache_layer_names_for_llama}
layer_quant_config = kv_cache_quant_config.copy()

# Define algorithm config by config file.
LLAMA_AUTOSMOOTHQUANT_CONFIG_FILE = './amd_quark-0.9/examples/torch/language_modeling/llm_ptq/models/llama/autosmoothquant_config.json'
algo_config = load_quant_algo_config_from_file(LLAMA_AUTOSMOOTHQUANT_CONFIG_FILE)

EXCLUDE_LAYERS = ["lm_head"]
quant_config = Config(
    global_quant_config=global_quant_config,
    layer_quant_config=layer_quant_config,
    kv_cache_quant_config=kv_cache_quant_config,
    exclude=EXCLUDE_LAYERS,
    algo_config=algo_config)
```

<a id="4-quantize-the-model"></a>

### 4. 量化模型
接下来，应用 量化。量化后，在导出之前冻结量化模型。

```python
### Quantization 
import torch
from quark.torch import ModelQuantizer
from quark.torch.export import JsonExporterConfig

# Apply quantization.
quantizer = ModelQuantizer(quant_config)
quant_model = quantizer.quantize_model(model, calib_dataloader)

# Freeze quantized model to export.
freezed_model = quantizer.freeze(model)
```

<a id="5-export-the-model"></a>

### 5.导出模型
使用 HuggingFace safetensors 格式导出模型。有关格式导出的更多详细信息，请参阅 HuggingFace safetensors [文档](https://github.com/huggingface/safetensors)。

```python
### Model Exporting
from quark.torch.quantization.config.config import Config
from quark.torch.export.config.config import ExporterConfig
from quark.shares.utils.log import ScreenLogger
from quark.torch import ModelExporter
from transformers import AutoTokenizer
from torch import nn
from pathlib import Path
from typing import List, Optional, Dict, Any
import torch
import json
import sys
import os

logger = ScreenLogger(__name__)
# Define export config.
LLAMA_KV_CACHE_GROUP = ["*k_proj", "*v_proj"]
export_config = ExporterConfig(json_export_config=JsonExporterConfig())
export_config.json_export_config.kv_cache_group = LLAMA_KV_CACHE_GROUP
export_path= "/workspace/models/Llama-3.3-70B-Instruct-MXFP4"


#EXPORT_DIR = MODEL_ID.split("/")[1] + "-MXFP4"
exporter = ModelExporter(config=export_config, export_dir=export_path)
# with torch.no_grad():
#     exporter.export_safetensors_model(freezed_model,quant_config=quant_config, tokenizer=tokenizer)

model = exporter.get_export_model(freezed_model, quant_config=quant_config, custom_mode="quark", add_export_info_for_hf=True)
model.save_pretrained(export_path)
try:
    # TODO: Having trust_remote_code=True by default in our codebase is dangerous.
    model_type = 'llama'
    use_fast = True if model_type in ["grok", "cohere", "olmo"] else False
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=use_fast)
    tokenizer.save_pretrained(export_path)
except Exception as e:
    logger.error(f"An error occurred when saving tokenizer: {e}.  You can try to save the tokenizer manually")
exporter.reset_model(model=model)
logger.info(f"hf_format quantized model exported to {export_path} successfully.")
```

<a id="6-infer-the-quantized-model-in-vllm"></a>

### 6. 推断 vLLM 中的量化模型
您现在可以直接通过 LLM 入口点加载并运行 Quark 量化模型：

```python
### Inference
from vllm import LLM, SamplingParams
import gc
import torch

def run(export_path: str):
    llm = LLM(
        model=export_path,
        kv_cache_dtype="fp8",
        quantization="quark",
        gpu_memory_utilization=0.8,   # mem usage limitation
    )
    return llm

if __name__ == "__main__":

    export_path = "/workspace/models/Llama-3.3-70B-Instruct-MXFP4"

    # Initialize LLM
    llm = run(export_path)
    print("LLM initialized.")

    # Input prompts
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Sampling parameters
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    print("Sampling params ready.")

    # Run inference
    outputs = llm.generate(prompts, sampling_params)
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)

# Release GPU memory
    del llm
    gc.collect()
    if torch.version.hip:   # ROCm backend
        torch.cuda.empty_cache()
        
```

<a id="quark-quantization-script"></a>

## Quark 量化 脚本
除了上面显示的Python API示例之外，Quark还提供了一个脚本来更方便地量化大型语言模型。它支持具有多种不同量化方案和优化算法的量化模型，并且可以导出量化模型并即时运行评估任务。使用脚本，上面的示例如下所示（您可以使用 `--output_dir` 选项更改输出目录）。运行此命令之前，请确保当前工作目录为 `./amd_quark-0.9/examples/torch/language_modeling/llm_ptq/`。

```python
import os
os.chdir("./amd_quark-0.9/examples/torch/language_modeling/llm_ptq/")
!python3 quantize_quark.py --model_dir /workspace/models/Llama-3.3-70B-Instruct  \
                          --model_attn_implementation "sdpa" \
                          --dataset /workspace/data/pile-val-backup \
                          --quant_scheme w_mxfp4_a_mxfp4 \
                          --group_size 32 \
                          --kv_cache_dtype fp8 \
                          --quant_algo autosmoothquant \
                          --min_kv_scale 1.0 \
                          --model_export hf_format \
                          --output_dir /workspace/models/Llama-3.3-70B-Instruct-MXFP4 \
                          --multi_gpu
```

以下命令排除某些图层以保留原始格式。确保您当前的工作目录是 `./amd_quark-0.9/examples/torch/language_modeling/llm_ptq/`。

```python
exclude_layers="*lm_head *layers.0.mlp.down_proj"
!python3 quantize_quark.py --model_dir meta-llama/Llama-3.3-70B-Instruct \
                          --quant_scheme w_mxfp4_a_mxfp4 \
                          --exclude_layers $exclude_layers 
```

<a id="acceleration-evaluation"></a>

## 加速评估

使用 vLLM 基准脚本来评估 推理 速度改进：

```python
!vllm bench latency \
--model /workspace/models/Llama-3.3-70B-Instruct-MXFP4 \
--tokenizer /workspace/models/Llama-3.3-70B-Instruct-MXFP4 \
--dataset-name random \
--input-len 4096 \
--output-len 1024 \
--num-prompts 80 \
--tensor-parallel 1 
```

<a id="accuracy-evaluation"></a>

## 准确度评估

您还可以使用 `lm_eval` 命令来评估准确性：

```python
!lm_eval --model vllm \
  --model_args pretrained=/workspace/models/Llama-3.3-70B-Instruct-MXFP4,kv_cache_dtype='fp8',quantization='quark' \
  --tasks gsm8k
```
