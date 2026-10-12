> 原文作者：Xinyu Kang and Liz Li  
> [AMD 官方教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/torchtitan_llama3.html) · [固定版本源码](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/pretrain/torchtitan_llama3.ipynb)  
> 上游版本：`0071bfd4bd3bd9525405c09a6ae69c983b09312a` · 同步日期：2026-10-12  
> 翻译状态：机器初译，待逐篇人工校对；代码与命令保持原文，未在 AMD GPU 上运行验证。

[中文 Notebook](../../notebooks/pretrain/torchtitan_llama3.ipynb) · [英文原文](../../original/pretrain/torchtitan_llama3.md) · [中英对照](../../bilingual/pretrain/torchtitan_llama3.md)

---

<a id="pretrain-llama-31-8b-with-torchtitan"></a>

# 使用 torchtitan 训练 Llama-3.1 8B
**作者**：Xinyu Kang and Liz Li
**知识水平**：中级

本教程演示如何利用 torchtitan 在 AMD ROCm GPU 上预训练 Llama-3.1 8B 大语言模型 (LLM)。 Torchtitan 是使用原生 PyTorch 进行大规模 LLM 培训的概念验证。该存储库以干净、最小的代码库展示了 PyTorch 的最新分布式训练功能。 Torchtitan 是对其他大型 LLM 训练代码库（例如 Megatron、MegaBlocks、LLM Foundry 和 DeepSpeed）的补充，而不是替代。欲了解更多信息，请参阅[torchtitan 官方 GitHub 页面](https://github.com/pytorch/torchtitan)。

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
!pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm6.3/ --force-reinstall
!pip3 install ipywidgets
!git clone https://github.com/pytorch/torchtitan
!cd torchtitan && pip install -r requirements.txt && pip install .
# This note book is verified under torch==2.7.0.dev20250302+rocm6.3,  torchvision==0.22.0.dev20250302+rocm6.3, torchtitan==0.0.2
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
torchtitan              0.0.2

```

**⚠️重要：确保选择正确的内核**

如果验证过程失败，请确保为您的Notebook选择正确的 Jupyter 内核。
要更改内核，请执行以下步骤：

1. 进入 **内核** 菜单。
2. 选择**更改内核**。
3. 从列表中选择 `Python 3 (ipykernel)`。

**重要**：未能选择正确的内核可能会导致运行Notebook时出现意外问题。

<a id="5-provide-your-hugging-face-token"></a>

### 5. 提供您的拥抱令牌

您需要 Hugging Face API 令牌才能访问 Llama-3.1。在 [Hugging Face Tokens](https://huggingface.co/settings/tokens) 生成您的令牌并请求访问 [Llama-3.1 8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)。令牌通常以“hf_”开头。 

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

<a id="pretrain-the-model"></a>

## 预训练模型

本节介绍使用 torchtitan 为 Llama-3.1 模型设置和运行 预训练 的过程。以下步骤描述了如何设置 GPU、导入所需的库、配置模型和训练参数以及运行 预训练 进程。

**⚠️重要：确保选择正确的内核**

确保为您的Notebook选择正确的 Jupyter 内核。要更改内核，请执行以下步骤：

1. 进入 **内核** 菜单。
2. 选择**更改内核**。
3. 从列表中选择 `Python 3 (ipykernel)`。

**重要**：未能选择正确的内核可能会导致运行Notebook时出现意外问题。

<a id="set-and-verify-the-gpu-availability"></a>

### 设置并验证 GPU 可用性

首先指定可用于 预训练 的 GPU。验证 PyTorch 是否正确检测到它们。

```python
import os
import torch
gpus = [0, 1, 2, 3, 4, 5, 6, 7] # Specify the 8 GPUs to be used for training for MI300x
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ','.join(map(str, gpus)))
# Ensure PyTorch detects the GPUs correctly
print(f"PyTorch detected number of available devices: {torch.cuda.device_count()}") 
```

<a id="download-the-llama-tokenizer"></a>

### 下载 Llama 分词器

 要开始训练 Llama-3.1 模型，您必须下载分词器模型。运行以下命令将 Llama-3.1 标记器下载到本地计算机。

```python
# Get your HF token from https://huggingface.co/settings/tokens

# Llama-3.1 tokenizer model
!cd torchtitan && python scripts/download_tokenizer.py --repo_id meta-llama/Meta-Llama-3.1-8B --tokenizer_path "original"
```

<a id="prepare-the-dataset"></a>

### 准备数据集

 默认情况下，torchtitan Llama-3.1 8B 使用 [c4 数据集](https://huggingface.co/datasets/allenai/c4) 进行 预训练。在此Notebook中，您可以将其更改为较小的 c4_test 数据集以进行测试。

```python
!sed -i 's/dataset = "c4"/dataset = "c4_test"/' ./torchtitan/torchtitan/models/llama/train_configs/llama3_8b.toml
```

<a id="run-the-pretrain-recipe"></a>

### 运行预训练配方

您可以使用以下命令在八个 GPU 上使用 LoRA 预训练 Llama-3.1 8B：

```python
!cd ./torchtitan/ && CONFIG_FILE="./torchtitan/models/llama/train_configs/llama3_8b.toml" ./run_train.sh
```

对于多节点分布式训练，torchtitan 提供了 Slurm 配方。您可以使用 `multinode_trainer.slurm` 文件提交 `sbatch` 作业。更多信息请参见torchtitan的官方[指南](https://github.com/pytorch/torchtitan?tab=readme-ov-file#multi-node-training)。

<a id="monitoring-gpu-memory"></a>

### 监控 GPU 内存

要在训练期间监控 GPU 内存，请在终端中运行以下命令。

**注意**：对于 ROCm 6.4 及更早版本，请改用 `rocm-smi` 命令。

```python
!amd-smi
```

此命令显示内存使用情况和其他 GPU 指标，以确保您的硬件资源得到最佳利用。
