> 原文作者：Xinyu Kang and Liz Li  
> [AMD 官方教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/torchtitan_llama3.html) · [固定版本源码](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/pretrain/torchtitan_llama3.ipynb)  
> 上游版本：`0071bfd4bd3bd9525405c09a6ae69c983b09312a` · 同步日期：2026-10-12  
> 翻译状态：机器初译，待逐篇人工校对；代码与命令保持原文，未在 AMD GPU 上运行验证。

<!-- cell:1 -->

<a id="pretrain-llama-31-8b-with-torchtitan"></a>

# 使用 torchtitan 训练 Llama-3.1 8B
**作者**：Xinyu Kang and Liz Li
**知识水平**：中级

本教程演示如何利用 torchtitan 在 AMD ROCm GPU 上预训练 Llama-3.1 8B 大语言模型 (LLM)。 Torchtitan 是使用原生 PyTorch 进行大规模 LLM 培训的概念验证。该存储库以干净、最小的代码库展示了 PyTorch 的最新分布式训练功能。 Torchtitan 是对其他大型 LLM 训练代码库（例如 Megatron、MegaBlocks、LLM Foundry 和 DeepSpeed）的补充，而不是替代。欲了解更多信息，请参阅[torchtitan 官方 GitHub 页面](https://github.com/pytorch/torchtitan)。

<details markdown="1">
<summary>查看英文原文 · 单元 1</summary>

# Pretrain Llama-3.1 8B with torchtitan

**Authors**: Xinyu Kang and Liz Li  
**Knowledge level**: Intermediate

This tutorial demonstrates how to pretrain the Llama-3.1 8B large language model (LLM) on AMD ROCm GPUs by leveraging torchtitan. Torchtitan is a proof of concept for large-scale LLM training using native PyTorch. The repository showcases PyTorch's latest distributed training features in a clean, minimal codebase. Torchtitan is complementary to, and not a replacement for, other large-scale LLM training codebases such as Megatron, MegaBlocks, LLM Foundry, and DeepSpeed. For more information, see the [official torchtitan GitHub page](https://github.com/pytorch/torchtitan).

</details>

<!-- cell:2 -->

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

<details markdown="1">
<summary>查看英文原文 · 单元 2</summary>

## Prerequisites

This tutorial was developed and tested using the following setup. 

### Operating system

* **Ubuntu 22.04**: Ensure your system is running Ubuntu version 22.04.

### Hardware

* **AMD Instinct™ GPUs**: This tutorial was tested on an AMD Instinct MI300X GPU. Ensure you are using an AMD Instinct GPU or compatible hardware with ROCm support and that your system meets the [official requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

### Software

* **ROCm 6.3**: Install and verify ROCm by following the [ROCm install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html). After installation, confirm your setup using:

    ``` bash
    amd-smi
    ```

    This command lists your AMD GPUs with relevant details.
    
    **Note**: For ROCm 6.4 and earlier, use the `rocm-smi` command instead.

* **Docker**: Ensure Docker is installed and configured correctly. Follow the Docker installation guide for your operating system.

   **Note**: Ensure the Docker permissions are correctly configured. To configure permissions to allow non-root access, run the following commands:

   ``` bash
   sudo usermod -aG docker $USER
   newgrp docker
   ```

   Verify Docker is working correctly:

   ``` bash
   docker run hello-world
   ```

### Hugging Face API access

* Obtain an API token from [Hugging Face](https://huggingface.co) for downloading models.
* Ensure the Hugging Face API token has the necessary permissions and approval to access the [Meta Llama checkpoints](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).

### Data preparation

* This tutorial uses a sample dataset from Hugging Face, which is prepared during the setup steps.

</details>

<!-- cell:3 -->

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

<details markdown="1">
<summary>查看英文原文 · 单元 3</summary>

## Prepare the training environment

### 1. Pull the Docker image

Ensure your system meets the [System Requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

Pull the Docker image required for this tutorial:

``` bash
docker pull rocm/pytorch-training:latest
```

### 2. Launch the Docker container

Launch the Docker container and map the necessary directories. Replace `/path/to/notebooks` with the full path to the directory on your host machine where these notebooks are stored.

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

**Note**: This command mounts the current directory to the `/workspace` directory in the container. Ensure the notebook file is either copied to this directory before running the Docker command or uploaded into the Jupyter Notebook environment after it starts. Save the token or URL provided in the terminal output to access the notebook from your web browser. You can download this notebook from the [AI Developer Hub GitHub repository](https://github.com/ROCm/gpuaidev).

### 3. Install and launch Jupyter

Inside the Docker container, install Jupyter using the following command:

``` bash
pip install jupyter
```

Start the Jupyter server:

``` bash
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**Note**: Ensure port `8888` is not already in use on your system before running the above command. If it is, you can specify a different port by replacing `--port=8888` with another port number, for example, `--port=8890`.

</details>

<!-- cell:4 -->

<a id="4-install-the-required-libraries"></a>

### 4.安装所需的库

安装本教程所需的库。在 Docker 容器中运行的 Jupyter Notebook 中运行以下命令：

<details markdown="1">
<summary>查看英文原文 · 单元 4</summary>

### 4. Install the required libraries

Install the libraries required for this tutorial. Run the following commands inside the Jupyter notebook running within the Docker container:

</details>

```python
!pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm6.3/ --force-reinstall
!pip3 install ipywidgets
!git clone https://github.com/pytorch/torchtitan
!cd torchtitan && pip install -r requirements.txt && pip install .
# This note book is verified under torch==2.7.0.dev20250302+rocm6.3,  torchvision==0.22.0.dev20250302+rocm6.3, torchtitan==0.0.2
```

<!-- cell:6 -->

验证安装：

<details markdown="1">
<summary>查看英文原文 · 单元 6</summary>

Verify the installation:

</details>

```python
# Verify the installation and version of the required libraries
!pip list | grep torch
```

<!-- cell:8 -->

这是预期的输出：

```
pytorch-triton-rocm     3.2.0+git4b3bb1f8
torch                   2.7.0.dev20250302+rocm6.3
torchdata               0.11.0
torchtitan              0.0.2

```

<details markdown="1">
<summary>查看英文原文 · 单元 8</summary>

Here is the expected output:

```
pytorch-triton-rocm     3.2.0+git4b3bb1f8
torch                   2.7.0.dev20250302+rocm6.3
torchdata               0.11.0
torchtitan              0.0.2

```

</details>

<!-- cell:9 -->

**⚠️重要：确保选择正确的内核**

如果验证过程失败，请确保为您的Notebook选择正确的 Jupyter 内核。
要更改内核，请执行以下步骤：

1. 进入 **内核** 菜单。
2. 选择**更改内核**。
3. 从列表中选择 `Python 3 (ipykernel)`。

**重要**：未能选择正确的内核可能会导致运行Notebook时出现意外问题。

<details markdown="1">
<summary>查看英文原文 · 单元 9</summary>

**⚠️ Important: ensure the correct kernel is selected**

If the verification process fails, ensure the correct Jupyter kernel is selected for your notebook.
To change the kernel, follow these steps:

1. Go to the **Kernel** menu.
2. Select **Change Kernel**.
3. Select `Python 3 (ipykernel)` from the list.

**Important**: Failure to select the correct kernel can lead to unexpected issues when running the notebook.

</details>

<!-- cell:10 -->

<a id="5-provide-your-hugging-face-token"></a>

### 5. 提供您的拥抱令牌

您需要 Hugging Face API 令牌才能访问 Llama-3.1。在 [Hugging Face Tokens](https://huggingface.co/settings/tokens) 生成您的令牌并请求访问 [Llama-3.1 8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)。令牌通常以“hf_”开头。 

在 Jupyter Notebook中运行以下交互式块来设置令牌：

**注意**：取消选中“添加令牌作为 Git 凭证”选项。

<details markdown="1">
<summary>查看英文原文 · 单元 10</summary>

### 5. Provide your Hugging Face token

You'll require a Hugging Face API token to access Llama-3.1. Generate your token at [Hugging Face Tokens](https://huggingface.co/settings/tokens) and request access for [Llama-3.1 8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B). Tokens typically start with "hf_". 

Run the following interactive block in your Jupyter notebook to set up the token:

**Note**: Uncheck the "Add token as Git credential" option.

</details>

```python
from huggingface_hub import notebook_login, HfApi

# Prompt the user to log in
notebook_login()
```

<!-- cell:12 -->

验证您的令牌是否被正确接受：

<details markdown="1">
<summary>查看英文原文 · 单元 12</summary>

Verify that your token was accepted correctly:

</details>

```python
from huggingface_hub import HfApi

try:
    api = HfApi()
    user_info = api.whoami()
    print(f"Token validated successfully! Logged in as: {user_info['name']}")
except Exception as e:
    print(f"Token validation failed. Error: {e}")
```

<!-- cell:14 -->

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

<details markdown="1">
<summary>查看英文原文 · 单元 14</summary>

## Pretrain the model

This section covers the process of setting up and running pretraining for the Llama-3.1 model using torchtitan. The following steps describe how to set up GPUs, import the required libraries, configure the model and training parameters, and run the pretraining process.

**⚠️ Important: ensure the correct kernel is selected**

Ensure the correct Jupyter kernel is selected for your notebook. To change the kernel, follow these steps:

1. Go to the **Kernel** menu.
2. Select **Change Kernel**.
3. Select `Python 3 (ipykernel)` from the list.

**Important**: Failure to select the correct kernel can lead to unexpected issues when running the notebook.

### Set and verify the GPU availability

Begin by specifying the GPUs that are available for pretraining. Verify that they are properly detected by PyTorch.

</details>

```python
import os
import torch
gpus = [0, 1, 2, 3, 4, 5, 6, 7] # Specify the 8 GPUs to be used for training for MI300x
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ','.join(map(str, gpus)))
# Ensure PyTorch detects the GPUs correctly
print(f"PyTorch detected number of available devices: {torch.cuda.device_count()}") 
```

<!-- cell:16 -->

<a id="download-the-llama-tokenizer"></a>

### 下载 Llama 分词器

 要开始训练 Llama-3.1 模型，您必须下载分词器模型。运行以下命令将 Llama-3.1 标记器下载到本地计算机。

<details markdown="1">
<summary>查看英文原文 · 单元 16</summary>

### Download the Llama tokenizer

 To start training Llama-3.1 models, you must download a tokenizer model. Run the following command to download the Llama-3.1 tokenizer to your local machine.

</details>

```python
# Get your HF token from https://huggingface.co/settings/tokens

# Llama-3.1 tokenizer model
!cd torchtitan && python scripts/download_tokenizer.py --repo_id meta-llama/Meta-Llama-3.1-8B --tokenizer_path "original"
```

<!-- cell:18 -->

<a id="prepare-the-dataset"></a>

### 准备数据集

 默认情况下，torchtitan Llama-3.1 8B 使用 [c4 数据集](https://huggingface.co/datasets/allenai/c4) 进行 预训练。在此Notebook中，您可以将其更改为较小的 c4_test 数据集以进行测试。

<details markdown="1">
<summary>查看英文原文 · 单元 18</summary>

### Prepare the dataset

 By default, torchtitan Llama-3.1 8B uses the [c4 dataset](https://huggingface.co/datasets/allenai/c4) to conduct pretraining. In this notebook, you can change this to the smaller c4_test dataset for test purposes.

</details>

```python
!sed -i 's/dataset = "c4"/dataset = "c4_test"/' ./torchtitan/torchtitan/models/llama/train_configs/llama3_8b.toml
```

<!-- cell:20 -->

<a id="run-the-pretrain-recipe"></a>

### 运行预训练配方

您可以使用以下命令在八个 GPU 上使用 LoRA 预训练 Llama-3.1 8B：

<details markdown="1">
<summary>查看英文原文 · 单元 20</summary>

### Run the pretrain recipe

You can pretrain Llama-3.1 8B with LoRA on eight GPUs using the following command:

</details>

```python
!cd ./torchtitan/ && CONFIG_FILE="./torchtitan/models/llama/train_configs/llama3_8b.toml" ./run_train.sh
```

<!-- cell:22 -->

对于多节点分布式训练，torchtitan 提供了 Slurm 配方。您可以使用 `multinode_trainer.slurm` 文件提交 `sbatch` 作业。更多信息请参见torchtitan的官方[指南](https://github.com/pytorch/torchtitan?tab=readme-ov-file#multi-node-training)。

<details markdown="1">
<summary>查看英文原文 · 单元 22</summary>

For multinode distributed training, torchtitan provides a Slurm recipe. You can use the `multinode_trainer.slurm` file to submit your `sbatch` job. For more information, see the official [guide](https://github.com/pytorch/torchtitan?tab=readme-ov-file#multi-node-training) from torchtitan.

</details>

<!-- cell:23 -->

<a id="monitoring-gpu-memory"></a>

### 监控 GPU 内存

要在训练期间监控 GPU 内存，请在终端中运行以下命令。

**注意**：对于 ROCm 6.4 及更早版本，请改用 `rocm-smi` 命令。

<details markdown="1">
<summary>查看英文原文 · 单元 23</summary>

### Monitoring GPU memory

To monitor GPU memory during training, run the following command in a terminal.

**Note**: For ROCm 6.4 and earlier, use the `rocm-smi` command instead.

</details>

```python
!amd-smi
```

<!-- cell:25 -->

此命令显示内存使用情况和其他 GPU 指标，以确保您的硬件资源得到最佳利用。

<details markdown="1">
<summary>查看英文原文 · 单元 25</summary>

This command displays memory usage and other GPU metrics to ensure your hardware resources are being optimally used.

</details>
