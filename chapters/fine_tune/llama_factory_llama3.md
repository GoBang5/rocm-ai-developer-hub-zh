> 原文作者：Alex He  
> [AMD 官方教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/fine_tune/llama_factory_llama3.html) · [固定版本源码](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/fine_tune/llama_factory_llama3.ipynb)  
> 上游版本：`0071bfd4bd3bd9525405c09a6ae69c983b09312a` · 同步日期：2026-10-12  
> 翻译状态：机器初译，待逐篇人工校对；代码与命令保持原文，未在 AMD GPU 上运行验证。

[中文 Notebook](../../notebooks/fine_tune/llama_factory_llama3.ipynb) · [英文原文](../../original/fine_tune/llama_factory_llama3.md) · [中英对照](../../bilingual/fine_tune/llama_factory_llama3.md)

---

<a id="fine-tune-llama-3-8b-with-llama-factory"></a>

# 使用 Llama-Factory 微调 Llama-3.1 8B
**作者**：Alex He
**知识水平**：中级

[![在 AMD Developer Cloud 中打开](../../original/assets/jupyter_notebook_button_fnl_small.png)](https://notebooks.amd.com/tutorials/llama_factory_llama3.ipynb)

本教程演示如何利用 Llama-Factory 在 AMD ROCm GPU 上微调 Llama-3 8B 大语言模型 (LLM)。高效的 微调 对于使大型语言模型 (LLM) 适应下游任务至关重要。然而，在不同的模型上实现这些方法需要付出巨大的努力。

[Llama-Factory](https://github.com/hiyouga/LLaMA-Factory)是一个统一的框架，集成了一套前沿、高效的训练方法。

它提供了灵活定制超过 100 个 LLM 的 微调 的解决方案，无需通过内置 Web UI LLAMABOARD 进行编码。

<a id="prerequisites"></a>

## 前置条件

本教程是使用以下设置进行开发和测试的。 

<a id="operating-system"></a>

### 操作系统

* **Ubuntu 22.04**：确保您的系统运行的是 Ubuntu 版本 22.04。

<a id="hardware"></a>

### 硬件

* **AMD Instinct™ GPU**：本教程在 AMD Instinct MI300X GPU 上进行了测试。确保您使用的是 AMD Instinct GPU 或支持 ROCm 的兼容硬件，并且您的系统满足[官方要求](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)。

<a id="software"></a>

### 软件

* **ROCm 6.3**：按照 [ROCm 安装指南](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html) 安装并验证 ROCm。安装后，使用以下命令确认您的设置：

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
* 确保 Hugging Face API 令牌具有访问 [Meta Llama 检查点](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) 所需的权限和批准。

<a id="data-preparation"></a>

### 数据准备

本教程使用 Hugging Face 中的示例数据集，该数据集是在设置步骤中准备的。

<a id="prepare-the-training-environment"></a>

## 准备训练环境

<a id="1-pull-the-docker-image"></a>

### 1. 拉取 Docker 镜像

确保您的系统满足[系统要求](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)。

拉取本教程所需的 Docker 映像：

``` bash
docker pull rocm/pytorch-training:latest
```

<a id="2-launch-the-docker-container"></a>

### 2.启动Docker容器

启动 Docker 容器并映射必要的目录。将 `/path/to/notebooks` 替换为主机上存储这些Notebook的目录的完整路径。

``` bash
docker run -it --rm \
  --network=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size 8G \
  -v $(pwd):/workspace \
  -w /workspace/notebooks \
  rocm/pytorch-training:latest
```

**注意**：该命令将当前目录挂载到容器中的`/workspace`目录中。确保在运行 Docker 命令之前将Notebook文件复制到此目录，或者在启动后上传到 Jupyter Notebook 环境。保存终端输出中提供的令牌或 URL，以便从 Web 浏览器访问Notebook。您可以从 [AI Developer Hub GitHub 存储库](https://github.com/ROCm/gpuaidev) 下载此Notebook。

<a id="3-install-and-launch-jupyter"></a>

### 3.安装并启动 Jupyter

在 Docker 容器内，使用以下命令安装 Jupyter：

``` bash
pip install jupyter
```

启动 Jupyter 服务器：

``` bash
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**注意**：在运行上述命令之前，请确保您的系统上尚未使用端口 `8888`。如果是，您可以通过将 `--port=8888` 替换为其他端口号来指定其他端口，例如 `--port=8890`。

<a id="4-install-the-required-libraries-and-frameworks"></a>

### 4. 安装所需的库和框架

运行下面的单元来安装本教程所需的库和框架。

* **DeepSpeed** – 通过减少内存使用并实现高效的分布式训练来优化大型模型训练。
* **LLaMA-Factory** – 一个开源框架，可简化大型语言模型 (LLM) 的培训和 微调。

```python
!pip3 install -U deepspeed==0.16.5
# Install Llama-Factory from source, pinned to commit a0d44c65 (0.9.4.dev0):
# the first commit with examples/deepspeed/, and still requires-python >=3.9
# so it installs on this container's Python 3.10.
!rm -rf LLaMA-Factory
!mkdir LLaMA-Factory && cd LLaMA-Factory && git init -q && git remote add origin https://github.com/hiyouga/LLaMA-Factory.git && git fetch -q --depth 1 origin a0d44c650a9a38c1ec6c8fc48a7daeacca9900a9 && git checkout -q FETCH_HEAD
!cd LLaMA-Factory && pip install -e ".[torch,metrics]"
# This notebook is verified under llamafactory==0.9.3.dev0 deepspeed==0.16.5
```

验证安装：

```python
# Verify the installation and version of the required libraries
!pip list | grep llamafactory
```

这是预期的输出：

```
llamafactory            0.9.4.dev0 
```

<a id="5-verify-llama-factory"></a>

### 5. 验证 Llama-Factory 

要确认软件包是否正确安装，请运行以下命令：

```python
!llamafactory-cli help
```

您应该看到以下输出：

```
----------------------------------------------------------------------
| Usage:                                                             |
|   llamafactory-cli api -h: launch an OpenAI-style API server       |
|   llamafactory-cli chat -h: launch a chat interface in CLI         |
|   llamafactory-cli eval -h: evaluate models                        |
|   llamafactory-cli export -h: merge LoRA adapters and export model |
|   llamafactory-cli train -h: train models                          |
|   llamafactory-cli webchat -h: launch a chat interface in Web UI   |
|   llamafactory-cli webui: launch LlamaBoard                        |
|   llamafactory-cli version: show version info                      |
----------------------------------------------------------------------
```

**⚠️重要**：确保选择正确的内核

如果验证过程失败，请确保为您的Notebook选择正确的 Jupyter 内核。
要更改内核，请执行以下步骤：

1. 进入 **内核** 菜单。
2. 选择**更改内核**。
3. 从列表中选择 `Python 3 (ipykernel)`。

**重要**：未能选择正确的内核可能会导致运行Notebook时出现意外问题。

<a id="6-provide-your-hugging-face-token"></a>

### 6. 提供您的拥抱令牌

您需要 Hugging Face API 令牌才能访问 Llama-3.1。在 [Hugging Face Tokens](https://huggingface.co/settings/tokens) 生成您的令牌并请求访问 [Llama-3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)。令牌通常以“hf_”开头。 

在 Jupyter Notebook中运行以下交互式块来设置令牌：

**注意**：取消选中“添加令牌作为 Git 凭证”选项。

```python
from huggingface_hub import notebook_login, HfApi

# Prompt the user to log in
notebook_login()
```

验证您的令牌是否被正确接受：

```python
from huggingface_hub import HfApi

try:
    api = HfApi()
    user_info = api.whoami()
    print(f"Token validated successfully! Logged in as: {user_info['name']}")
except Exception as e:
    print(f"Token validation failed. Error: {e}")
```

<a id="fine-tuning-the-model"></a>

## 微调 型号

本节介绍使用 Llama-Factory 为 Llama-3 模型设置和运行 微调 的过程。以下步骤描述了如何设置 GPU、导入所需的库、配置模型和训练参数以及运行 微调 进程。

**⚠️重要**：确保选择正确的内核

确保为您的Notebook选择正确的 Jupyter 内核。要更改内核，请执行以下步骤：

1. 进入 **内核** 菜单。
2. 选择**更改内核**。
3. 从列表中选择 `Python 3 (ipykernel)`。

**重要**：未能选择正确的内核可能会导致运行Notebook时出现意外问题。

<a id="set-and-verify-the-gpu-availability"></a>

### 设置并验证 GPU 可用性

首先指定可用于 微调 的 GPU。验证 PyTorch 是否正确检测到它们。

```python
import os
import torch
gpus= [0, 1] # Rank 0 is for MI300x single device finetune, and Rank 0/1 for full 
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ','.join(map(str, gpus)))
# Ensure PyTorch detects the GPUs correctly
print(f"PyTorch detected number of available devices: {torch.cuda.device_count()}") 
```

<a id="download-the-model"></a>

### 下载模型

运行以下命令将权重下载到本地计算机。这还会下载分词器模型和负责任的使用指南。

要下载 Llama-3.1-8B-Instruct，请使用以下命令：

```python
!huggingface-cli download --resume-download meta-llama/Llama-3.1-8B-Instruct
```

<a id="run-the-fine-tuning-recipes"></a>

### 运行 微调 配方

现在使用 微调 配方 `llama3_full_sft.yaml` 进行分布式训练。此配方预定义了 [数据集标识](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/identity.json) 和 [alpaca_en_demo](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/alpaca_en_demo.json) JSON 文件。

**注意**：这些数据集仅用于评估目的。您可能需要将它们更改为实际的。

以下命令使用可用的 GPU 微调 Llama-3.1 8B 模型。
>下面的命令每 500 步保存一个 检查点。运行该命令之前，请根据需要在YAML文件中配置相应的`save_only_model`和`save_steps`值。

```python
!cd LLaMA-Factory/ && FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/llama3_full_sft.yaml model_name_or_path=meta-llama/Llama-3.1-8B-Instruct
```

<a id="monitoring-gpu-memory"></a>

### 监控 GPU 内存

要在训练期间监控 GPU 内存，请在终端中运行以下命令。

>**观看-n0 rocm-smi**

**注意**：对于 ROCm 6.5 及更高版本，请改用 `amd-smi` 命令。

此命令显示内存使用情况和其他 GPU 指标，以确保您的硬件资源得到最佳利用。

<a id="conclusion"></a>

### 结论

LLaMA-Factory + DeepSpeed + ROCm 为 AMD GPU 上的 微调 LLM 提供了简单高效的工作流程。
