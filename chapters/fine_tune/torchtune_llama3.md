> 原文作者：Bill He  
> [AMD 官方教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/fine_tune/torchtune_llama3.html) · [固定版本源码](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/fine_tune/torchtune_llama3.ipynb)  
> 上游版本：`0071bfd4bd3bd9525405c09a6ae69c983b09312a` · 同步日期：2026-10-12  
> 翻译状态：机器初译，待逐篇人工校对；代码与命令保持原文，未在 AMD GPU 上运行验证。

[中文 Notebook](../../notebooks/fine_tune/torchtune_llama3.ipynb) · [英文原文](../../original/fine_tune/torchtune_llama3.md) · [中英对照](../../bilingual/fine_tune/torchtune_llama3.md)

---

<a id="fine-tune-llama-31-8b-with-torchtune"></a>

# 使用 torchtune 微调 Llama-3.1 8B
**作者**：Bill He
**知识水平**：中级

本教程演示如何利用 torchtune 在 AMD ROCm GPU 上微调 Llama-3.1 8B 大语言模型 (LLM)。 Torchtune 是一个易于使用的 PyTorch 库，用于 LLM 的创作、后期培训和实验。其特点：

- SFT、知识蒸馏、RL 和 RLHF 以及 量化 感知培训的可破解培训方案。
- 流行 LLM 的简单 PyTorch 实现，如 Llama、Gemma、Mistral、Phi、Qwen 等。
- 利用最新的 PyTorch API 实现 OOTB 一流的内存效率、性能改进和扩展。
- YAML 配置可轻松配置培训、评估、量化 或 推理 配方。

有关更多信息，请参阅[torchtune 官方 GitHub 页面](https://github.com/pytorch/torchtune)。

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

* 本教程使用 Hugging Face 中的示例数据集，该数据集是在设置步骤中准备的。

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

<a id="4-install-the-required-libraries"></a>

### 4.安装所需的库

安装本教程所需的库。在 Docker 容器中运行的 Jupyter Notebook 中运行以下命令：

```python
# Install PyTorch, torchvision, torchao nightlies
!pip install --pre --upgrade torch torchvision torchao --index-url https://download.pytorch.org/whl/nightly/rocm6.3/
!pip install --pre --upgrade torchtune --extra-index-url https://download.pytorch.org/whl/nightly/rocm6.3/ 
# This note book is verified under torch==2.7.0.dev20250302+rocm6.3, torchao==0.10.0.dev20250303+rocm6.3,  torchvision==0.22.0.dev20250302+rocm6.3, torchtune==0.6.0.dev20250302+rocm6.3
```

验证安装：

```python
# Verify the installation and version of the required libraries
!pip list | grep torch
```

这是预期的输出：

```
pytorch-triton-rocm     3.2.0+git4b3bb1f8
torch                   2.7.0.dev20250302+rocm6.3
torchdata               0.11.0
torchtune               0.6.0.dev20250302+rocm6.3
```

<a id="5-verify-torchtune-for-rocm-63"></a>

### 5. 验证 ROCm 6.3 的 torchtune

要确认软件包是否已正确安装，请运行以下命令：

```python
!tune --help
```

您应该看到以下输出：

```
usage: tune [-h] {download,ls,cp,run,validate,cat} ...

Welcome to the torchtune CLI!

options:
  -h, --help            show this help message and exit

subcommands:
  {download,ls,cp,run,validate,cat}
    download            Download a model from the Hugging Face Hub or Kaggle
                        Model Hub.
    ls                  List all built-in recipes and configs
    cp                  Copy a built-in recipe or config to a local path.
    run                 Run a recipe. For distributed recipes, this supports
                        all torchrun arguments.
    validate            Validate a config and ensure that it is well-formed.
    cat                 Pretty print a config, making it easy to know which
                        parameters you can override with `tune run`.
```

**⚠️重要：确保选择正确的内核**

如果验证过程失败，请确保为您的Notebook选择正确的 Jupyter 内核。
要更改内核，请执行以下步骤：

1. 进入 **内核** 菜单。
2. 选择**更改内核**。
3. 从列表中选择 `Python 3 (ipykernel)`。

**重要**：未能选择正确的内核可能会导致运行Notebook时出现意外问题。

<a id="6-provide-your-hugging-face-token"></a>

### 6. 提供您的拥抱令牌

您需要 Hugging Face API 令牌才能访问 Llama-3.1。在 [Hugging Face Tokens](https://huggingface.co/settings/tokens) 生成您的令牌并请求访问 [Llama-3.1 8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)。令牌通常以“hf_”开头。 

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

本节介绍使用 torchtune 为 Llama-3.1 模型设置和运行 微调 的过程。以下步骤描述了如何设置 GPU、导入所需的库、配置模型和训练参数以及运行 微调 进程。

**⚠️重要：确保选择正确的内核**

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

<a id="download-the-llama-model"></a>

### 下载 Llama 模型

运行以下命令将权重下载到本地计算机。这还会下载分词器模型和负责任的使用指南。

要下载 Llama-3.1，请使用以下命令：

```python
!tune download meta-llama/Meta-Llama-3.1-8B-Instruct \
    --output-dir /tmp/Meta-Llama-3.1-8B-Instruct \
    --ignore-patterns "original/consolidated.00.pth"
```

<a id="run-the-fine-tuning-recipes"></a>

### 运行 微调 配方

使用这些 微调 配方进行单 GPU 或分布式训练。

<a id="single-gpu-training"></a>

#### 单 GPU 训练

默认情况下，微调 使用 [alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) 数据集，您可以使用以下命令在单个 GPU 上使用 LoRA 微调 Llama-3.1 8B：

```python
!tune run lora_finetune_single_device --config llama3_1/8B_lora_single_device
```

<a id="distributed-training"></a>

#### 分布式训练
对于分布式训练，tune CLI 与 torchrun 集成。

**注意**：您必须有多个 GPU 才能运行分布式训练示例。要在两个 GPU 上运行 Llama-3.1 8B 的完整微调，请使用以下命令：

```python
!tune run --nproc_per_node 2 full_finetune_distributed --config llama3_1/8B_full
```

<a id="customize-your-recipes-for-llama"></a>

### 为 Llama 定制您的食谱

修改配置有两种方法。

<a id="configuration-overrides"></a>

#### 配置覆盖

您可以直接从命令行覆盖配置字段。例如，您可以将 批大小 设置为 `16` 并使用相同的命令禁用激活检查点：

```python
!tune run lora_finetune_single_device \
    --config llama3_1/8B_lora_single_device \
    batch_size=16 \
    enable_activation_checkpointing=False 
```

**注意**：如果遇到内存不足 (OOM) 错误，请减少 `batch_size` 或启用梯度检查点。使用 `amd-smi` 监视 微调 期间 VRAM 使用情况。对于 ROCm 6.4 及更早版本，请使用 `rocm-smi` 命令来实现此目的。

<a id="update-a-local-copy"></a>

#### 更新本地副本

您也可以将配置复制到本地目录并直接修改内容：

```python
!tune cp llama3_1/8B_lora_single_device ./my_custom_config_llama3_1_8B_lora_single_device.yaml
# Copied to ./my_custom_config_llama3_1_8B_lora_single_device.yaml
```

然后，您可以通过将 `tune run` 命令应用于本地文件来运行自定义配方：

```python
!tune run lora_finetune_single_device --config ./my_custom_config_llama3_1_8B_lora_single_device.yaml
```

运行 `tune --help` 命令以查看所有可能的 CLI 命令和选项。有关使用和更新配置的更多信息，请参阅官方 torchtune [deep-dive](https://meta-pytorch.org/torchtune/main/deep_dives/configs.html)。

<a id="custom-datasets"></a>

### 自定义数据集

`torchtune` 在各种不同的数据集上支持 微调，包括指令风格、聊天风格、偏好数据集等。要详细了解如何应用这些组件来微调您自己的自定义数据集，请参阅提供的链接以及 torchtune API [文档](https://meta-pytorch.org/torchtune/main/api_ref_datasets.html)。

<a id="monitoring-gpu-memory"></a>

### 监控 GPU 内存

要在训练期间监控 GPU 内存，请在终端中运行以下命令。

**注意**：对于 ROCm 6.4 及更早版本，请改用 `rocm-smi` 命令。

```python
!amd-smi
```

此命令显示内存使用情况和其他 GPU 指标，以确保您的硬件资源得到最佳利用。
