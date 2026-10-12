> 原文作者：Shekhar Pandey and Liz Li  
> [AMD 官方教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/train_llama_mock_data.html) · [固定版本源码](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/pretrain/train_llama_mock_data.ipynb)  
> 上游版本：`0071bfd4bd3bd9525405c09a6ae69c983b09312a` · 同步日期：2026-10-12  
> 翻译状态：机器初译，待逐篇人工校对；代码与命令保持原文，未在 AMD GPU 上运行验证。

[中文 Notebook](../../notebooks/pretrain/train_llama_mock_data.ipynb) · [英文原文](../../original/pretrain/train_llama_mock_data.md) · [中英对照](../../bilingual/pretrain/train_llama_mock_data.md)

---

<a id="training-llama-31-8b-with-megatron-lm"></a>

# 使用 Megatron-LM 训练大语言模型
**作者**：Shekhar Pandey and Liz Li
**知识水平**：中级

本教程演示如何使用*模拟数据*训练 Llama-3.1 模型。 Llama-3.1 8B 模型是一种流行的开源 大语言模型 (LLM)，旨在高效处理各种自然语言处理任务。要了解有关 Llama 模型的更多信息，请访问 [Llama 网站](https://www.llama.com/)。

本教程使用模拟数据提供训练工作流程的快速、轻量级演示，使您能够验证您的环境是否配置正确且功能正常。模拟数据是一种无需大型数据集即可验证训练流程的有用方法。

训练过程利用了 Megatron-LM 框架，这是一个针对 预训练 和 微调 大规模语言模型的专用框架。有关 Megatron-LM 的更多信息，请参阅他们的 [GitHub 存储库](https://github.com/NVIDIA/Megatron-LM)。所有步骤都在 Docker 容器中执行，该容器提供了具有所有必要依赖项的即用型环境。

本教程以 [预训练 with Megatron-LM 教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/setup_tutorial.html) 中完成的设置为基础。

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

* **ROCm 6.2**：按照 [ROCm 安装指南](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html) 安装并验证 ROCm。安装后，使用以下命令确认您的设置：

    ``` bash
    amd-smi
    ```

    此命令列出了您的 AMD GPU 以及相关详细信息。
    
    **注意**：对于 ROCm 6.4 及更早版本，请改用 `rocm-smi` 命令。

* **Docker**：确保 Docker 已正确安装和配置。请遵循适用于您的操作系统的 Docker 安装指南。

   **注意**：确保正确配置 Docker 权限。要配置权限以允许非 root 访问，请运行以下命令：

   ```bash
   sudo usermod -aG docker $USER
   newgrp docker
   ```

   验证 Docker 是否正常工作：

   ```bash
   docker run hello-world
   ```

<a id="hugging-face-api-access"></a>

### Hugging Face API 访问

* 从[Hugging Face](https://huggingface.co)获取API令牌用于下载模型。
* 确保 Hugging Face API 令牌具有访问 [Meta 的 Llama 检查点](https://huggingface.co/meta-llama/Llama-3.1-8B) 所需的权限和批准。

<a id="prepare-the-training-environment"></a>

## 准备训练环境

当您的系统满足 前置条件 要求后，请按照以下步骤设置培训环境。

<a id="1-pull-the-docker-image"></a>

### 1. 拉取 Docker 镜像

在终端中运行以下命令来提取预构建的 Docker 映像。 Docker 镜像提供了所有必要的依赖项，包括 PyTorch、PyTorch Lightning、ROCm 库和 Megatron-LM 实用程序。

``` bash
docker pull rocm/megatron-lm:24.12-dev
```

<a id="2-launch-the-docker-container"></a>

### 2.启动Docker容器

在终端中运行以下命令以启动具有适当配置的 Docker 容器：

```bash
docker run -it --rm \
  --device /dev/dri \
  --device /dev/kfd \
  --network host \
  --ipc host \
  --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --privileged \
  --name megatron-dev-env \
  -v $(pwd):/workspace \
  -w /workspace/notebooks \
  rocm/megatron-lm:24.12-dev \
  /bin/bash
```

**注意**：该命令将当前目录挂载到容器中的`/workspace`目录中。确保在运行命令之前将Notebook文件复制到此目录，或者在启动后将Notebook文件上传到 Jupyter Notebook 环境。保存终端输出中提供的令牌或 URL，以便从 Web 浏览器访问Notebook。您可以从 [AI Developer Hub GitHub 存储库](https://github.com/ROCm/gpuaidev) 下载此Notebook。

<a id="3-install-jupyter-and-start-the-server"></a>

### 3.安装Jupyter并启动服务器

在 Docker 容器内，使用以下命令安装 Jupyter：

```bash
pip install jupyter
```

启动 Jupyter 服务器：

```bash
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```
**注意**：在运行上述命令之前，请确保端口 `8888` 尚未在您的系统上使用。如果是，请通过将 `--port=8888` 替换为其他端口号来指定其他端口，例如 `--port=8890`。

<a id="4-access-the-jupyter-notebook"></a>

### 4.访问JupyterNotebook

有两种方法可以加载 Jupyter Notebook：

<a id="mount-the-notebook-with-docker"></a>

#### 使用 Docker 挂载Notebook

* **确保正确挂载**：启动容器时，使用 `-v /path/to/notebooks:/workspace/notebooks` 选项挂载您的Notebook目录。
* **在容器中找到Notebook**：在 Jupyter 界面内，导航到 `/workspace/notebooks` 以查找您的文件并开始处理它们。

####通过浏览器上传Notebook

* **启动 Docker 时跳过 `-v` 选项**：如果您不想挂载Notebook目录，请省略 `-v /path/to/notebooks:/workspace/notebooks` 选项。
* **手动上传**：在浏览器中访问Jupyter界面后：
  1. 单击右上角的**上传**按钮。
  2. 从主机中选择您的Notebook文件并将其上传到容器。
* **开始工作**：Notebook上传后，单击Notebook文件名将其打开并开始工作。

<a id="5-clone-the-megatron-lm-repository"></a>

### 5. 克隆 Megatron-LM 存储库

在 Docker 容器内运行以下命令来克隆 Megatron-LM 存储库并导航到经过验证的提交：

```python
# Clone the Megatron-LM repository and navigate to the validated commit
!git clone https://github.com/ROCm/Megatron-LM && cd Megatron-LM && git checkout bb93ccbfeae6363c67b361a97a27c74ab86e7e92
```

<a id="6-provide-your-hugging-face-token"></a>

### 6. 提供您的拥抱令牌

可以通过登录您的帐户 [Hugging Face Tokens](https://huggingface.co/settings/tokens) 来生成 Hugging Face 令牌。

您需要 Hugging Face API 令牌才能访问 Llama-3.1 8B。在 Hugging Face Tokens 生成您的令牌并请求 Llama-3.1 8B 的访问权限。令牌通常以“hf_”开头。

在 Jupyter Notebook中运行以下交互式块来设置令牌：

**注意**：取消选中“添加令牌作为 Git 凭证”选项。

```python
from huggingface_hub import notebook_login, HfApi

# Prompt the user to log in
notebook_login()
```

验证您的令牌是否被正确接受：

```python
try:
    api = HfApi()
    user_info = api.whoami()
    print(f"Token validated successfully! Logged in as: {user_info['name']}")
except Exception as e:
    print(f"Token validation failed. Error: {e}")
```

<a id="run-the-training-script"></a>

## 运行训练脚本

本节介绍如何运行训练脚本，并对关键参数进行解释。

<a id="single-node-training-overview"></a>

### 单节点训练概述

训练过程涉及运行预配置的脚本，该脚本初始化并执行 Llama-3.1 模型的训练。该脚本利用 Megatron-LM 框架和模拟数据来模拟完整的训练流程。此方法可确保您的环境配置正确并且适用于实际用例。

运行脚本之前，请确保所有环境变量均已正确设置。

<a id="key-parameters-for-training"></a>

### 训练的关键参数：

* **批大小 (`BS`)**：将此设置为 `64` 以获得最佳 GPU 使用率。
* **序列长度(`SEQ_LENGTH`)**：输入序列长度，设置为`4096`。
* **张量并行 (`TP`)**：将其设置为 `8` 以获得高效的并行性。
* **精度 (`TE_FP8`)**：将其设置为 `0` 以获得 `BF16` 精度。

<a id="run-the-training-script-1"></a>

### 运行训练脚本

使用以下命令在单个节点上训练模型：

```python
!cd Megatron-LM && TEE_OUTPUT=1 MBS=2 BS=64 TP=8 TE_FP8=0 SEQ_LENGTH=4096  \
TOKENIZER_MODEL='meta-llama/Llama-3.1-8B' MODEL_SIZE='8' \
bash examples/llama/train_llama3.sh
```

<a id="additional-details-about-the-command"></a>

### 有关命令的其他详细信息

该命令使用以下参数配置训练过程：

* **`TEE_OUTPUT=1`**：启用日志输出到控制台。
* **`MBS=2`**：每个 GPU 为 Micro-批大小。
* **`BS=64`**：所有 GPU 上的总 批大小。
* **`TP=8`**：张量并行 用于跨 GPU 分发模型。
* **`TE_FP8=0`**：将精度设置为 `BF16` 进行训练。
* **`SEQ_LENGTH=4096`**：最大输入序列长度。

训练脚本执行以下操作：
* 使用模拟数据作为输入。
* 使用指定配置训练 Llama-3.1 8B 模型。

您可以根据您的硬件和所需的配置，通过修改命令详细信息来自定义这些参数。

<a id="monitor-the-training-progress"></a>

## 监控训练进度

在训练过程中监视输出日志的以下发展：

* **迭代进度**：已完成的迭代次数。
* **损失值**：这表明模型的学习进度。较低的值表明更好的学习。
* **GPU 利用率**：确保硬件资源的最佳使用。

日志打印到控制台并保存到脚本指定目录中的日志文件中。

<a id="key-notes"></a>

## 要点

* 模拟数据仅用于验证。要使用不同的数据集，请参阅 [预训练 with Megatron-LM 教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/setup_tutorial.html)。
* 根据您的硬件调整超参数。本教程中的超参数配置基于 8 个 MI300x GPU 的一个节点。
* 此示例说明如何在单个节点上运行训练任务。有关多节点训练说明，请参阅[预训练 with Megatron-LM 教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/setup_tutorial.html)。
* 验证日志的正确性。
