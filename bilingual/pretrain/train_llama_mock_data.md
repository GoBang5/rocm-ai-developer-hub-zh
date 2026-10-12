> 原文作者：Shekhar Pandey and Liz Li  
> [AMD 官方教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/train_llama_mock_data.html) · [固定版本源码](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/pretrain/train_llama_mock_data.ipynb)  
> 上游版本：`0071bfd4bd3bd9525405c09a6ae69c983b09312a` · 同步日期：2026-10-12  
> 翻译状态：机器初译，待逐篇人工校对；代码与命令保持原文，未在 AMD GPU 上运行验证。

<!-- cell:1 -->

<a id="training-llama-31-8b-with-megatron-lm"></a>

# 使用 Megatron-LM 训练大语言模型
**作者**：Shekhar Pandey and Liz Li
**知识水平**：中级

本教程演示如何使用*模拟数据*训练 Llama-3.1 模型。 Llama-3.1 8B 模型是一种流行的开源 大语言模型 (LLM)，旨在高效处理各种自然语言处理任务。要了解有关 Llama 模型的更多信息，请访问 [Llama 网站](https://www.llama.com/)。

本教程使用模拟数据提供训练工作流程的快速、轻量级演示，使您能够验证您的环境是否配置正确且功能正常。模拟数据是一种无需大型数据集即可验证训练流程的有用方法。

训练过程利用了 Megatron-LM 框架，这是一个针对 预训练 和 微调 大规模语言模型的专用框架。有关 Megatron-LM 的更多信息，请参阅他们的 [GitHub 存储库](https://github.com/NVIDIA/Megatron-LM)。所有步骤都在 Docker 容器中执行，该容器提供了具有所有必要依赖项的即用型环境。

本教程以 [预训练 with Megatron-LM 教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/setup_tutorial.html) 中完成的设置为基础。

<details markdown="1">
<summary>查看英文原文 · 单元 1</summary>

# Training Llama-3.1 8B with Megatron-LM 

**Authors**: Shekhar Pandey and Liz Li  
**Knowledge level**: Intermediate

This tutorial demonstrates how to train the Llama-3.1 model using *mock data*. The Llama-3.1 8B model is a popular open-source large language model (LLM) designed to handle a wide range of natural language processing tasks efficiently. Learn more about the Llama models at [Llama's website](https://www.llama.com/).

This tutorial uses mock data to provide a quick and lightweight demonstration of the training workflow, enabling you to verify that your environment is correctly configured and functional. Mock data is a useful way to validate the training pipeline without requiring large datasets.

The training process leverages the Megatron-LM framework, a specialized framework for pretraining and fine-tuning large-scale language models. For more information about Megatron-LM, see their [GitHub repository](https://github.com/NVIDIA/Megatron-LM). All steps are executed within a Docker container, which provides a ready-to-use environment with all necessary dependencies.

This tutorial builds on the setup completed in the [Pretraining with Megatron-LM tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/setup_tutorial.html).

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

<details markdown="1">
<summary>查看英文原文 · 单元 2</summary>

## Prerequisites

This tutorial was developed and tested using the following setup. 

### Operating system

* **Ubuntu 22.04**: Ensure your system is running Ubuntu version 22.04.

### Hardware

* **AMD Instinct™ GPUs**: This tutorial was tested on an AMD Instinct MI300X GPU. Ensure you are using an AMD Instinct GPU or compatible hardware with ROCm support and that your system meets the [official requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

### Software

* **ROCm 6.2**: Install and verify ROCm by following the [ROCm install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html). After installation, confirm your setup using:

    ``` bash
    amd-smi
    ```

    This command lists your AMD GPUs with relevant details.
    
    **Note**: For ROCm 6.4 and earlier, use the `rocm-smi` command instead.

* **Docker**: Ensure Docker is installed and configured correctly. Follow the Docker installation guide for your operating system.

   **Note**: Ensure the Docker permissions are correctly configured. To configure permissions to allow non-root access, run the following commands:

   ```bash
   sudo usermod -aG docker $USER
   newgrp docker
   ```

   Verify Docker is working correctly:

   ```bash
   docker run hello-world
   ```

### Hugging Face API access

* Obtain an API token from [Hugging Face](https://huggingface.co) for downloading models.
* Ensure the Hugging Face API token has the necessary permissions and approval to access [Meta's Llama checkpoints](https://huggingface.co/meta-llama/Llama-3.1-8B).

</details>

<!-- cell:3 -->

<a id="prepare-the-training-environment"></a>

## 准备训练环境

当您的系统满足 前置条件 要求后，请按照以下步骤设置培训环境。

<details markdown="1">
<summary>查看英文原文 · 单元 3</summary>

## Prepare the training environment

After your system meets the prerequisites, follow these steps to set up the training environment.

</details>

<!-- cell:4 -->

<a id="1-pull-the-docker-image"></a>

### 1. 拉取 Docker 镜像

在终端中运行以下命令来提取预构建的 Docker 映像。 Docker 镜像提供了所有必要的依赖项，包括 PyTorch、PyTorch Lightning、ROCm 库和 Megatron-LM 实用程序。

``` bash
docker pull rocm/megatron-lm:24.12-dev
```

<details markdown="1">
<summary>查看英文原文 · 单元 4</summary>

### 1. Pull the Docker image

Run the following command in your terminal to pull the prebuilt Docker image. The Docker image provides all necessary dependencies, including PyTorch, PyTorch Lightning, ROCm libraries, and Megatron-LM utilities.

``` bash
docker pull rocm/megatron-lm:24.12-dev
```

</details>

<!-- cell:5 -->

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

<details markdown="1">
<summary>查看英文原文 · 单元 5</summary>

### 2. Launch the Docker container

Run the following command in your terminal to launch the Docker container with the appropriate configuration:

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

**Note**: This command mounts the current directory to the `/workspace` directory in the container. Ensure the notebook file is either copied to this directory before running the command or uploaded into the Jupyter Notebook environment after it starts. Save the token or URL provided in the terminal output to access the notebook from your web browser. You can download this notebook from the [AI Developer Hub GitHub repository](https://github.com/ROCm/gpuaidev).

</details>

<!-- cell:6 -->

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

<details markdown="1">
<summary>查看英文原文 · 单元 6</summary>

### 3. Install Jupyter and start the server

Inside the Docker container, install Jupyter using the following command:

```bash
pip install jupyter
```

Start the Jupyter server:

```bash
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```
**Note**: Ensure port `8888` is not already in use on your system before running the above command. If it is, specify a different port by replacing `--port=8888` with another port number, for example, `--port=8890`.

</details>

<!-- cell:7 -->

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

<details markdown="1">
<summary>查看英文原文 · 单元 7</summary>

### 4. Access the Jupyter notebook

There are two ways to load your Jupyter notebook:

#### Mount the notebook with Docker

* **Ensure the correct mounting**: When launching the container, use the `-v /path/to/notebooks:/workspace/notebooks` option to mount your notebook directory.
* **Locate the notebook in the container**: Inside the Jupyter interface, navigate to `/workspace/notebooks` to find your files and begin working on them.

#### Upload the notebook through the browser

* **Skip the `-v` option when launching Docker**: If you prefer not to mount your notebook directory, omit the `-v /path/to/notebooks:/workspace/notebooks` option.
* **Upload manually**: After accessing the Jupyter interface in your browser:
  1. Click the **Upload** button in the top-right corner.
  2. Select your notebook file from your host machine and upload it to the container.
* **Start working**: After the notebook is uploaded, click the notebook filename to open it and begin working.

</details>

<!-- cell:8 -->

<a id="5-clone-the-megatron-lm-repository"></a>

### 5. 克隆 Megatron-LM 存储库

在 Docker 容器内运行以下命令来克隆 Megatron-LM 存储库并导航到经过验证的提交：

<details markdown="1">
<summary>查看英文原文 · 单元 8</summary>

### 5. Clone the Megatron-LM repository

Run the following commands inside the Docker container to clone the Megatron-LM repository and navigate to the validated commit:

</details>

```python
# Clone the Megatron-LM repository and navigate to the validated commit
!git clone https://github.com/ROCm/Megatron-LM && cd Megatron-LM && git checkout bb93ccbfeae6363c67b361a97a27c74ab86e7e92
```

<!-- cell:10 -->

<a id="6-provide-your-hugging-face-token"></a>

### 6. 提供您的拥抱令牌

可以通过登录您的帐户 [Hugging Face Tokens](https://huggingface.co/settings/tokens) 来生成 Hugging Face 令牌。

您需要 Hugging Face API 令牌才能访问 Llama-3.1 8B。在 Hugging Face Tokens 生成您的令牌并请求 Llama-3.1 8B 的访问权限。令牌通常以“hf_”开头。

在 Jupyter Notebook中运行以下交互式块来设置令牌：

**注意**：取消选中“添加令牌作为 Git 凭证”选项。

<details markdown="1">
<summary>查看英文原文 · 单元 10</summary>

### 6. Provide your Hugging Face token

A Hugging Face token can be generated by signing into your account at [Hugging Face Tokens](https://huggingface.co/settings/tokens).

You'll require a Hugging Face API token to access Llama-3.1 8B. Generate your token at Hugging Face Tokens and request access for Llama-3.1 8B. Tokens typically start with "hf_".

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
try:
    api = HfApi()
    user_info = api.whoami()
    print(f"Token validated successfully! Logged in as: {user_info['name']}")
except Exception as e:
    print(f"Token validation failed. Error: {e}")
```

<!-- cell:14 -->

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

<details markdown="1">
<summary>查看英文原文 · 单元 14</summary>

## Run the training script

This section describes how to run the training script, with an explanation of the key parameters.

### Single-node training overview

The training process involves running a pre-configured script that initializes and executes the training of the Llama-3.1 model. The script leverages the Megatron-LM framework and mock data to simulate a full training pipeline. This approach ensures your environment is configured correctly and is functional for real-world use cases.

Before running the script, ensure all environment variables are set correctly.

### Key parameters for training:

* **Batch size (`BS`)**: Set this to `64` for optimal GPU usage.
* **Sequence length (`SEQ_LENGTH`)**: Input sequence length, set to `4096`.
* **Tensor parallelism (`TP`)**: Set this to `8` for efficient parallelism.
* **Precision (`TE_FP8`)**: Set this to `0` for `BF16` precision.

### Run the training script

Use the following command to train the model on a single node:

</details>

```python
!cd Megatron-LM && TEE_OUTPUT=1 MBS=2 BS=64 TP=8 TE_FP8=0 SEQ_LENGTH=4096  \
TOKENIZER_MODEL='meta-llama/Llama-3.1-8B' MODEL_SIZE='8' \
bash examples/llama/train_llama3.sh
```

<!-- cell:16 -->

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

<details markdown="1">
<summary>查看英文原文 · 单元 16</summary>

### Additional details about the command

This command configures the training process with the following parameters:

* **`TEE_OUTPUT=1`**: Enables logging output to the console.
* **`MBS=2`**: Micro-batch size per GPU.
* **`BS=64`**: Total batch size across all GPUs.
* **`TP=8`**: Tensor parallelism for distributing the model across GPUs.
* **`TE_FP8=0`**: Sets the precision to `BF16` for training.
* **`SEQ_LENGTH=4096`**: Maximum input sequence length.

The training script does the following:
* Uses mock data as input.
* Trains the Llama-3.1 8B model with the specified configurations.

You can customize these parameters based on your hardware and desired configurations by modifying the command details.

</details>

<!-- cell:17 -->

<a id="monitor-the-training-progress"></a>

## 监控训练进度

在训练过程中监视输出日志的以下发展：

* **迭代进度**：已完成的迭代次数。
* **损失值**：这表明模型的学习进度。较低的值表明更好的学习。
* **GPU 利用率**：确保硬件资源的最佳使用。

日志打印到控制台并保存到脚本指定目录中的日志文件中。

<details markdown="1">
<summary>查看英文原文 · 单元 17</summary>

## Monitor the training progress

Monitor the output logs during the training process for the following developments:

* **Iteration progress**: The number of completed iterations.
* **Loss values**: This indicates the model's learning progress. Lower values suggest better learning.
* **GPU utilization**: Ensures the optimal usage of your hardware resources.

Logs are printed to the console and saved to a log file within the directory specified by the script.

</details>

<!-- cell:18 -->

<a id="key-notes"></a>

## 要点

* 模拟数据仅用于验证。要使用不同的数据集，请参阅 [预训练 with Megatron-LM 教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/setup_tutorial.html)。
* 根据您的硬件调整超参数。本教程中的超参数配置基于 8 个 MI300x GPU 的一个节点。
* 此示例说明如何在单个节点上运行训练任务。有关多节点训练说明，请参阅[预训练 with Megatron-LM 教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/setup_tutorial.html)。
* 验证日志的正确性。

<details markdown="1">
<summary>查看英文原文 · 单元 18</summary>

## Key notes

* Mock data is for validation only. To use a different dataset, see the [Pretraining with Megatron-LM tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/setup_tutorial.html).
* Tune the hyperparameters based on your hardware. The hyperparameter configuration in this tutorial is based on one node of 8x MI300x GPUs.
* This example illustrates how to run a training task on a single node. For multi-node training instructions, see the [Pretraining with Megatron-LM tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/setup_tutorial.html).
* Verify the logs for correctness.

</details>
