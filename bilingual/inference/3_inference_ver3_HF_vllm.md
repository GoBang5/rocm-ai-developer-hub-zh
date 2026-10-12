> 原文作者：Seungrok Jung and Hyukjoon Lee  
> [AMD 官方教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/3_inference_ver3_HF_vllm.html) · [固定版本源码](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/inference/3_inference_ver3_HF_vllm.ipynb)  
> 上游版本：`0071bfd4bd3bd9525405c09a6ae69c983b09312a` · 同步日期：2026-10-12  
> 翻译状态：机器初译，待逐篇人工校对；代码与命令保持原文，未在 AMD GPU 上运行验证。

<!-- cell:1 -->

<a id="deploying-llama-31-8b-using-vllm"></a>

# 使用 vLLM 部署模型
**作者**：Seungrok Jung and Hyukjoon Lee
**知识水平**：初学者

vLLM 是一个开源库，旨在为 大语言模型 (LLM) 推理 提供高 吞吐量 和低 延迟。它通过高效批处理请求和充分利用 GPU 资源来优化文本生成工作负载，使开发人员能够管理代码生成和大规模对话式 AI 等复杂任务。

本教程将指导您使用 ROCm 软件堆栈在 AMD Instinct™ GPU 上设置和运行 vLLM。了解如何配置您的环境、容器化您的工作流程以及将测试查询发送到 vLLM 支持的 推理 服务器。

<details markdown="1">
<summary>查看英文原文 · 单元 1</summary>

# Deploying Llama-3.1 8B using vLLM

**Authors**: Seungrok Jung and Hyukjoon Lee  
**Knowledge level**: Beginner

vLLM is an open-source library designed to deliver high throughput and low latency for large language model (LLM) inference. It optimizes text generation workloads by efficiently batching requests and making full use of GPU resources, empowering developers to manage complex tasks like code generation and large-scale conversational AI.

This tutorial guides you through setting up and running vLLM on AMD Instinct™ GPUs using the ROCm software stack. Learn how to configure your environment, containerize your workflow, and send test queries to the vLLM-supported inference server.

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

* **AMD Instinct GPU**：本教程在 AMD Instinct MI300X GPU 上进行了测试。确保您使用的是 AMD Instinct GPU 或支持 ROCm 的兼容硬件，并且您的系统满足[官方要求](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)。

<a id="software"></a>

### 软件

* **ROCm 6.2 或 6.3**：按照 [ROCm 安装指南](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html) 安装并验证 ROCm。安装后，使用以下命令确认您的设置：

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
* 确保 Hugging Face API 令牌具有访问 [Meta 的 Llama 检查点](https://huggingface.co/meta-llama/Llama-3.1-8B) 所需的权限和批准。

<details markdown="1">
<summary>查看英文原文 · 单元 2</summary>

## Prerequisites

This tutorial was developed and tested using the following setup. 

### Operating system

* **Ubuntu 22.04**: Ensure your system is running Ubuntu version 22.04.

### Hardware

* **AMD Instinct GPUs**: This tutorial was tested on an AMD Instinct MI300X GPU. Ensure you are using an AMD Instinct GPU or compatible hardware with ROCm support and that your system meets the [official requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

### Software

* **ROCm 6.2 or 6.3**: Install and verify ROCm by following the [ROCm install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html). After installation, confirm your setup using:

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
* Ensure the Hugging Face API token has the necessary permissions and approval to access [Meta's Llama checkpoints](https://huggingface.co/meta-llama/Llama-3.1-8B).

</details>

<!-- cell:3 -->

<a id="prepare-the-inference-environment"></a>

## 准备推理环境

按照以下步骤使 推理 环境做好使用准备。

<a id="1-launch-the-docker-container"></a>

### 1.启动Docker容器

在终端中运行以下命令以提取包含所有必要依赖项的预构建 Docker 映像，并使用正确的配置启动 Docker 容器：

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
  --hostname=ROCm-FT \
  --env HUGGINGFACE_HUB_CACHE=/workspace \
  -v $(pwd):/workspace \
  -w /workspace/notebooks \
  --entrypoint /bin/bash \
  vllm/vllm-openai-rocm:v0.15.0
```

**注意**：该命令将当前目录挂载到容器中的`/workspace`目录中。确保在运行 Docker 命令之前将Notebook文件复制到此目录，或者在启动后上传到 Jupyter Notebook 环境。保存终端输出中提供的令牌或 URL，以便从 Web 浏览器访问Notebook。您可以从 [AI Developer Hub GitHub 存储库](https://github.com/ROCm/gpuaidev) 下载此Notebook。

<a id="2-install-and-launch-jupyter"></a>

### 2.安装并启动 Jupyter

在 Docker 容器内，使用以下命令安装 Jupyter：

``` bash
pip install jupyter
```

然后启动Jupyter服务器：

``` bash
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**注意**：在运行上述命令之前，请确保您的系统上尚未使用端口 `8888`。如果是，您可以通过将 `--port=8888` 替换为其他端口号来指定其他端口，例如 `--port=8890`。

<a id="3-provide-your-hugging-face-token"></a>

### 3. 提供您的拥抱令牌

您需要 Hugging Face API 令牌才能访问 meta-llama/Llama-3.1-8B-Instruct。在 Hugging Face Tokens 生成您的令牌并请求访问 [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)。令牌通常以“hf_”开头。

在 Jupyter Notebook中运行以下交互式块来设置令牌：

**注意**：取消选中“添加令牌作为 Git 凭证？”选项。

<details markdown="1">
<summary>查看英文原文 · 单元 3</summary>

## Prepare the inference environment

Follow these steps to get the inference environment ready for use.

### 1. Launch the Docker container

Run the following command in your terminal to pull the prebuilt Docker image containing all necessary dependencies and launch the Docker container with the proper configuration:

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
  --hostname=ROCm-FT \
  --env HUGGINGFACE_HUB_CACHE=/workspace \
  -v $(pwd):/workspace \
  -w /workspace/notebooks \
  --entrypoint /bin/bash \
  vllm/vllm-openai-rocm:v0.15.0
```

**Note**: This command mounts the current directory to the `/workspace` directory in the container. Ensure the notebook file is either copied to this directory before running the Docker command or uploaded into the Jupyter Notebook environment after it starts. Save the token or URL provided in the terminal output to access the notebook from your web browser. You can download this notebook from the [AI Developer Hub GitHub repository](https://github.com/ROCm/gpuaidev).

### 2. Install and launch Jupyter

Inside the Docker container, install Jupyter using the following command:

``` bash
pip install jupyter
```

Then start the Jupyter server:

``` bash
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**Note**: Ensure port `8888` is not already in use on your system before running the above command. If it is, you can specify a different port by replacing `--port=8888` with another port number, for example, `--port=8890`.

### 3. Provide your Hugging Face token

You'll require a Hugging Face API token to access meta-llama/Llama-3.1-8B-Instruct. Generate your token at Hugging Face Tokens and request access for [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct). Tokens typically start with "hf_".

Run the following interactive block in your Jupyter notebook to set up the token:

**Note**: Uncheck the "Add token as Git credential?" option.

</details>

```python
from huggingface_hub import notebook_login, HfApi

# Prompt the user to log in
notebook_login()
```

<!-- cell:5 -->

验证您的令牌是否被正确接受：

<details markdown="1">
<summary>查看英文原文 · 单元 5</summary>

Verify that your token was accepted correctly:

</details>

```python
# Validate the token
try:
    api = HfApi()
    user_info = api.whoami()
    print(f"Token validated successfully! Logged in as: {user_info['name']}")
except Exception as e:
    print(f"Token validation failed. Error: {e}")
```

<!-- cell:7 -->

<a id="deploying-the-llm-using-vllm"></a>

## 使用 vLLM 部署 LLM

开始在 Jupyter Notebook 中使用 vLLM 部署 LLM (meta-llama/Llama-3.1-8B-Instruct)：

<a id="start-the-vllm-server"></a>

### 启动 vLLM 服务器 

运行以下命令启动 vLLM 服务器：

<details markdown="1">
<summary>查看英文原文 · 单元 7</summary>

## Deploying the LLM using vLLM

Start deploying the LLM (meta-llama/Llama-3.1-8B-Instruct) using vLLM in the Jupyter notebook:

### Start the vLLM server 

Run this command to launch the vLLM server:

</details>

```python
!HIP_VISIBLE_DEVICES=2 python3 -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct
```

<!-- cell:9 -->

连接成功后显示`INFO: Uvicorn running on socket ('0.0.0.0', XX) (Press CTRL+C to quit)`。

**注意**：在多 GPU 环境中，建议设置 `HIP_VISIBLE_DEVICES=x` 在您的首选 GPU 上部署 LLM。

<a id="start-the-client"></a>

### 启动客户端

如上所述成功运行服务器后，打开一个新Notebook并向服务器发送查询，如下所示。

**注意**：对于此步骤，必须打开一个新的 Python Notebook。创建Notebook单元后，复制以下代码并在新Notebook中运行它。可以通过选择“**文件->新建->Notebook**”来打开新Notebook。

<details markdown="1">
<summary>查看英文原文 · 单元 9</summary>

After successfully connecting, it displays `INFO: Uvicorn running on socket ('0.0.0.0', XX) (Press CTRL+C to quit)`.

**Note**: In a multi-GPU environment, the setting `HIP_VISIBLE_DEVICES=x` is recommended to deploy the LLM on your preferred GPU.

### Start the client

After successfully running the server, as described above, open a new notebook and send a query to the server as shown below.

**Note**: For this step, a new Python notebook must be opened. After the notebook cell is created, copy the code below and run it in your new notebook. New notebooks can be opened by selecting **File->New->Notebook**.

</details>

```python
import requests

url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages": [
        {
            "role": "system",
            "content": "You are an expert in the field of AI. Make sure to provide an explanation in few sentences."
        },
        {
            "role": "user",
            "content": "Explain the concept of AI."
        }
    ],
    "stream": False,
    "max_tokens": 128
}

response = requests.post(url, headers=headers, json=data)
print(response.json())
```

<!-- cell:11 -->

如果连接成功，输出将是：

``` bash
{"id":"chat-xx","object":"chat.completion","created":1736494622,"model":"meta-llama/Meta-Llama-3.1-8B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"Artificial Intelligence (AI) is a field of computer science ...}
```

<details markdown="1">
<summary>查看英文原文 · 单元 11</summary>

If the connection is successful, the output will be:

``` bash
{"id":"chat-xx","object":"chat.completion","created":1736494622,"model":"meta-llama/Meta-Llama-3.1-8B-Instruct","choices":[{"index":0,"message":{"role":"assistant","content":"Artificial Intelligence (AI) is a field of computer science ...}
```

</details>
