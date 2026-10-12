> 原文作者：Shailen Sobhee  
> [AMD 官方教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/deepseek_janus_cpu_gpu.html) · [固定版本源码](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/inference/deepseek_janus_cpu_gpu.ipynb)  
> 上游版本：`0071bfd4bd3bd9525405c09a6ae69c983b09312a` · 同步日期：2026-10-12  
> 翻译状态：机器初译，待逐篇人工校对；代码与命令保持原文，未在 AMD GPU 上运行验证。

[中文 Notebook](../../notebooks/inference/deepseek_janus_cpu_gpu.ipynb) · [英文原文](../../original/inference/deepseek_janus_cpu_gpu.md) · [中英对照](../../bilingual/inference/deepseek_janus_cpu_gpu.md)

---

<a id="janus-pro-deepseek-model-on-amd-hardware"></a>

# 在 CPU 或 GPU 上运行 DeepSeek Janus Pro
**作者**：Shailen Sobhee
**知识水平**：初学者

本教程演示如何使用 Janus Pro 自回归框架执行多模态 推理，该框架是 DeepSeek AI 的 [Janus 系列](https://github.com/deepseek-ai/Janus#-janus-series-unified-multimodal-understanding-and-generation-models) 的一部分。您将在高性能 AMD 硬件上运行该模型，包括 EPYC™ CPU 和 Instinct™ GPU。

术语“多模态”意味着模型可以同时理解和处理来自多个来源的信息，例如文本和图像。通过统一这些不同的数据类型（称为模态），Janus 可以实现复杂的理解和生成任务。

本教程还介绍了在 CPU 上执行模型时如何利用 PyTorch 的 [AMD ZenDNN](https://www.amd.com/en/developer/zendnn.html) 插件（也称为 zentorch）来加速推理。

<a id="prerequisites"></a>

## 前置条件

使用以下设置来运行本教程。

<a id="hardware"></a>

### 硬件

对于本教程，您需要一个配备 AMD Instinct GPU 的系统。要在 CPU 上运行模型并使用 AMD ZenDNN，您需要 AMD EPYC CPU。 

本教程在以下硬件上进行了测试：
* AMD 本能 MI100
* AMD 本能 MI210
* AMD Instinct MI300X
* 第四代 AMD EPYC（热那亚）
* 第五代 AMD EPYC（都灵）

<a id="software"></a>

### 软件

* **Ubuntu 22.04**：确保您的系统运行的是 Ubuntu 22.04 或更高版本。
* **ROCm 6.3**：这仅是 GPU 执行所必需的。按照 [ROCm 安装指南](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html) 安装并验证 ROCm。
* **PyTorch 2.6**（或更高版本）
* **zentorch 5.0.2**（或更高版本）
* 克隆官方[Janus-Pro DeepSeek存储库](https://github.com/deepseek-ai/Janus)。
  
**注意**：本教程已使用 `torch2.7.1+rocm6.3`、`torch2.6.0+cpu` 和 `zentorch-5.0.2` 进行测试。

<a id="install-and-launch-jupyter-notebooks"></a>

### 安装并启动 Jupyter Notebooks
如果您的系统上尚未安装 Jupyter，请安装它并使用以下命令启动 JupyterLab：

```
pip install jupyter
```

要启动 Jupyter 服务器，请运行以下命令：

```
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**注意**：在运行上述命令之前，请确保您的系统上尚未使用端口 `8888`。如果是，您可以通过将 `--port=8888` 替换为其他端口号来指定其他端口，例如 `--port=8890`。

命令执行后，终端输出将显示 URL 和令牌。复制此 URL 并将其粘贴到主机上的 Web 浏览器中以访问 JupyterLab。启动 JupyterLab 后，将此Notebook上传到环境并继续按照本教程中的步骤进行操作。

<a id="prepare-the-environment-and-install-dependencies"></a>

## 准备环境并安装依赖

以下命令安装成功运行本教程所需的所有依赖项，以及 DeepSeek AI GitHub 存储库中的 `janus` 模块。

```python
!pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.3
!pip install "transformers<4.50" ipywidgets
!pip install git+https://github.com/deepseek-ai/Janus.git
```

接下来，快速检查 PyTorch 环境。验证 GPU 硬件是否可以使用 ROCm 后端进行访问。如果没有，则在 CPU 上执行模型。

```python
import torch 
print(f"PyTorch Version: {torch.__version__}")

print("--- GPU Verification ---")
if torch.cuda.is_available():
    print("✅ PyTorch has access to the GPU.")
    print(f"ROCm Version (detected by PyTorch): {torch.version.hip}")
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
    print(f"GPU installed on the system: {torch.cuda.get_device_name(0)}")
else:
    print("❌ PyTorch CANNOT access the GPU. Please check your ROCm installation and drivers or proceed to continue with executing on CPU")
```

首先导入本教程所需的 Python 库。

```python
import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
import time
```

为即将到来的推理进程初始化以下变量：

```python
iteration = 1
warmup = 0
max_new_tokens = 512
dtype = "bfloat16"
```

<a id="pipeline-summary"></a>

## 管道总结

- 选择硬件后端：CPU 或 GPU
- 加载 Janus-Pro 模型和处理器。
- 定义多模态理解的图像和问题。
- 预处理文本和图像输入。
- 为模型生成图像嵌入。
- 利用 zentorch。
- 运行预热迭代以稳定性能。
- 执行定时 推理 来测量 延迟。
- 计算并显示平均生成时间。
- 解码并显示人工智能生成的响应。

<a id="choose-the-hardware-backend-cpu-or-gpu"></a>

## 选择硬件后端：CPU 或 GPU

此示例演示了可以执行 AI 工作负载的两种可能的硬件后端：CPU 或 GPU。要将工作负载部署在 GPU 上，请设置 `device = "cuda"`。否则，要部署在 CPU 上，请设置 `device = "cpu"`。 CPU 支持多个软件后端，例如 `zentorch`、Intel® Extension for PyTorch (`ipex`) 和默认 PyTorch CPU 后端（也称为 `inductor`）。您还可以在本机模式下运行本教程（例如，急切模式而不是图形模式）。

**注意**：zentorch 插件版本 5.0.2 需要仅 CPU 版本的 PyTorch 2.6。这是 PyTorch 的限制。作为解决方法，请删除所有以前的 PyTorch 安装并安装 CPU 版本。

```python
device = "cuda" # Change to "cpu" to execute on CPU
backend = "zentorch" # Available CPU backends: zentorch, inductor, ipex, or native [Eager Mode]

if device == "cpu" and backend == "zentorch":
    !pip uninstall -y torch torchvision torchaudio pytorch-triton-rocm
    !pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
    !pip install zentorch #--no-cache-dir
    import torch 
    import zentorch
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Zentorch Version: {zentorch.__version__}")

amp_enabled = True if dtype != "float32" else False
amp_dtype = getattr(torch, dtype)
```

<a id="step-1-model-initialization-and-setup"></a>

## 步骤 1：模型初始化和设置

首先指定模型路径并初始化处理图像和文本所需的组件。

```python
# Specify the path to the model  
model_path = "deepseek-ai/Janus-Pro-7B"  

# Load the multimodal chat processor and tokenizer  
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)  
tokenizer = vl_chat_processor.tokenizer  

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(  
    model_path, trust_remote_code=True  
)
```

将模型转换为使用 `BFloat16` 精度，并将其移至 推理 的 CPU。

```python
vl_gpt = vl_gpt.to(amp_dtype).to(device).eval() 
```

<a id="step-2-define-the-image-and-user-input"></a>

## 步骤 2：定义图像和用户输入

定义图像和基于文本的查询来分析其内容。

```python
image = "../assets/deepseek_janus_demo_small.jpg" 
```

使用下面的代码片段检查您的图像。这也证实了图像路径是正确的。

```python
from IPython.display import Image
from IPython.core.display import HTML 
Image(image)
```

现在准备 视觉语言模型 的输入有效负载。这是通过构建符合模型所需聊天模板的 `conversation` 列表来实现的。用户消息是一个字典，其中包含角色、嵌入 `<image_placeholder>` 令牌的文本问题以及相应的图像对象。

```python
question = "What is happening in this image?"  

conversation = [  
    {  
        "role": "<|User|>",  
        "content": f"<image_placeholder>\n{question}",  
        "images": [image],  
    },   
]
```

<a id="step-3-preprocess-the-image-and-text-inputs"></a>

## 步骤 3：预处理图像和文本输入

加载图像并将对话数据转换为适合模型的格式。

```python
# Load images and prepare inputs  
pil_images = load_pil_images(conversation)  

# Process conversation and images into model-compatible input format  
prepare_inputs = vl_chat_processor(  
    conversations=conversation, images=pil_images, force_batchify=True  
).to(vl_gpt.device)  
```

<a id="step-4-generate-image-embeddings"></a>

## 步骤 4：生成图像嵌入

在运行 推理 之前，处理图像以获得其嵌入。

```python
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)  
```

<a id="step-5-leverage-the-amd-zendnn-plugin-for-pytorch-zentorch"></a>

## 步骤 5：利用 PyTorch (zentorch) 的 AMD ZenDNN 插件 

AMD 已将 zentorch 注册为 `torch.compile` 的自定义后端。该后端通过一个名为 `optimize()` 的函数在 AOT Autograd 之后集成了 ZenDNN 优化。此函数在 Aten IR 层的基于 FX 的图表上运行，以生成优化的基于 FX 的图表作为输出。有关该插件及其操作的更多信息，请参阅[ZenDNN 用户指南](https://docs.amd.com/r/en-US/57300-ZenDNN-user-guide/ZenDNN)。

```python
if device == "cpu":
    if(backend == "zentorch"):
        print("Backend: ZenTorch")
        import zentorch
        torch._dynamo.reset()
        vl_gpt.language_model.forward = torch.compile(vl_gpt.language_model.forward, backend="zentorch")  
    
    elif(backend == "inductor"):
        print("Backend: Inductor")
        torch._dynamo.reset()
        vl_gpt.language_model.forward = torch.compile(vl_gpt.language_model.forward)  
    
    else:
        print("Running in Eager mode")
else:
    print("We are executing on GPU therefore we won't be leveraging any CPU-acceleration software like Zentorch.")
```

<a id="profiler"></a>

### 分析器

PyTorch Profiler 有助于验证 `torch.compile` 的操作 (ops) 并评估其优化模型的有效性。它通过跟踪执行时间和查明可以进行优化的区域来深入了解模型的性能，确保 `torch.compile` 按预期工作。

如果您的重点是性能检查而不是详细分析，则可以跳过此部分。

```python
# # Start profiling
# from torch.profiler import profile, record_function, ProfilerActivity
# def trace_handler(prof):
#     # Print profiling information after each step
#     print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))

# with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=False,schedule=torch.profiler.schedule(wait=1, warmup=1, active=1),on_trace_ready=trace_handler,) as prof:
#     for i in range(3):
#     # Run the model to get the response
#         outputs = vl_gpt.language_model.generate(
#             inputs_embeds=inputs_embeds,
#             attention_mask=prepare_inputs.attention_mask,
#             pad_token_id=tokenizer.eos_token_id,
#             bos_token_id=tokenizer.bos_token_id,
#             eos_token_id=tokenizer.eos_token_id,
#             max_new_tokens=max_new_tokens,
#             do_sample=False,
#             use_cache=True,
#         )
#         prof.step()

# # To check the DataType
# for name, param in vl_gpt.named_parameters():
#     print(f"Parameter: {name}, Shape: {param.shape}, Data Type: {param.dtype}")
#     print(f"First few values: {param.flatten()[:5]}\n") 
```

<a id="step-6-warm-up-inference-stabilization-runs"></a>

## 步骤 6：热身 推理（稳定跑）

为了确保性能稳定，请运行几个 推理 循环而不测量时间。

```python
for i in range(warmup):  
    # Generate a response without timing for warmup  
    outputs = vl_gpt.language_model.generate(  
        inputs_embeds=inputs_embeds,  
        attention_mask=prepare_inputs.attention_mask,  
        pad_token_id=tokenizer.eos_token_id,  
        bos_token_id=tokenizer.bos_token_id,  
        eos_token_id=tokenizer.eos_token_id,
        min_new_tokens = max_new_tokens,  
        max_new_tokens = max_new_tokens,  
        do_sample=False,  
        use_cache=True,  
    )  
    print(f"WARMUP:{i+1} COMPLETED!")  
```

<a id="step-7-timed-inference-execution"></a>

## 步骤7：定时执行推理

现在运行实际的 推理，同时测量 延迟 进行性能分析。

```python
total_time = 0.0  

for i in range(iteration):  
    tic = time.time()  # Start time  

    # Generate response from the model  
    outputs = vl_gpt.language_model.generate(  
        inputs_embeds=inputs_embeds,  
        attention_mask=prepare_inputs.attention_mask,  
        pad_token_id=tokenizer.eos_token_id,  
        bos_token_id=tokenizer.bos_token_id,  
        eos_token_id=tokenizer.eos_token_id,  
        min_new_tokens = max_new_tokens,
        max_new_tokens = max_new_tokens,  
        do_sample=False,  
        use_cache=True,  
    )  

    toc = time.time()  # End time  
    delta = toc - tic  # Compute time taken  
    total_time = total_time + delta  
```

<a id="step-8-compute-and-display-latency"></a>

## 步骤8：计算并显示延迟

接下来，计算平均值 延迟 并打印结果。

```python
total_time = total_time / iteration
print( 
    f"e2e_latency (TTFT + Generation Time) for step: {total_time:.6f} sec", 
    flush=True, 
)

tps_per_step = (max_new_tokens / total_time)
print( 
    f"Throughput: {tps_per_step:.6f} tokens/sec", 
    flush=True, 
)
```

<a id="step-9-decode-and-display-the-output"></a>

## 步骤9：解码并显示输出

最后，将生成的令牌序列解码为人类可读的文本。

```python
answer = tokenizer.decode(outputs[0].to(device).tolist(), skip_special_tokens=True)  
print(f"{prepare_inputs['sft_format'][0]}", answer)  
```

<a id="pipeline-summary"></a>

## 管道总结

- 选择硬件后端：CPU 或 GPU  
- 加载 Janus-Pro 模型和处理器。  
- 定义多模态理解的图像和问题。  
- 预处理文本和图像输入。  
- 为模型生成图像嵌入。  
- 利用 zentorch。  
- 运行预热迭代以稳定性能。  
- 执行定时 推理 来测量 延迟。  
- 计算并显示平均生成时间。  
- 解码并显示人工智能生成的响应。
