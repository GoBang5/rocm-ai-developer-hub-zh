> 原文作者：Seungrok Jung and Hyukjoon Lee  
> [AMD 官方教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/1_inference_ver3_HF_transformers.html) · [固定版本源码](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/inference/1_inference_ver3_HF_transformers.ipynb)  
> 上游版本：`0071bfd4bd3bd9525405c09a6ae69c983b09312a` · 同步日期：2026-10-12  
> 翻译状态：机器初译，待逐篇人工校对；代码与命令保持原文，未在 AMD GPU 上运行验证。

<!-- cell:1 -->

<a id="running-inference-with-hugging-face-transformers"></a>

# 使用 Hugging Face Transformers 进行推理
**作者**：Seungrok Jung and Hyukjoon Lee
**知识水平**：初学者

本教程探讨如何在 AMD 硬件上利用 Hugging Face Transformers。了解如何为 AMD Instinct™ GPU 安装和配置 ROCm 并启动您最喜爱的型号。通过执行这些步骤，您将能够在 ROCm 加速环境中运行高级法学硕士，利用 AMD 的 GPU 性能来执行创新的自然语言处理任务。

<details markdown="1">
<summary>查看英文原文 · 单元 1</summary>

# Running inference with Hugging Face Transformers

**Authors**: Seungrok Jung and Hyukjoon Lee  
**Knowledge level**: Beginner

This tutorial explores how to leverage Hugging Face Transformers on AMD hardware. Learn how to install and configure ROCm for AMD Instinct™ GPUs and launch your favorite models. By following these steps, you’ll be able to run advanced LLMs in a ROCm-accelerated environment, capitalizing on AMD’s GPU performance for innovative natural language processing tasks.

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

<a id="inference-on-hugging-face-transformers"></a>

## 推理 Hugging Face变形金刚

Hugging Face Transformers 是一个流行的开源库，它提供了一个易于使用的界面，用于处理广泛使用的语言模型，例如 BERT、GPT 和 Llama 变体。这些模型可以进行微调，也可以现成用于文本生成、问题回答和情感分析等任务。   

本教程演示如何使用 AMD Instinct GPU 在 Hugging Face Transformers 模型上运行 推理。它涵盖配置 ROCm 以支持 GPU、安装必要的库以及在容器化环境中运行 LLM (meta-llama/Meta-Llama-3.1-8B-Instruct)。 

<a id="prepare-the-inference-environment"></a>

## 准备推理环境

要设置 推理 环境，请按照以下步骤操作。

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
  rocm/pytorch:rocm6.3.1_ubuntu22.04_py3.10_pytorch
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

<a id="3-install-the-required-libraries"></a>

### 3.安装所需的库

安装本教程所需的库。在 Docker 容器中运行的 Jupyter Notebook 中运行以下命令：

<details markdown="1">
<summary>查看英文原文 · 单元 3</summary>

## Inference on Hugging Face Transformers

Hugging Face Transformers is a popular open-source library that provides an easy-to-use interface for working with widely used language models, such as BERT, GPT, and the Llama variants. These models can be fine-tuned or used off-the-shelf for tasks like text generation, question answering, and sentiment analysis.   

This tutorial demonstrates how to run inference on Hugging Face Transformers models using AMD Instinct GPUs. It covers configuring ROCm for GPU support, installing the necessary libraries, and running an LLM (meta-llama/Meta-Llama-3.1-8B-Instruct) in a containerized environment. 

## Prepare the inference environment

To set up the inference environment, follow these steps.

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
  rocm/pytorch:rocm6.3.1_ubuntu22.04_py3.10_pytorch
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

### 3. Install the required libraries

Install the libraries needed for this tutorial. Run the following commands inside the Jupyter notebook running within the Docker container:

</details>

```python
!pip install accelerate transformers 
```

<!-- cell:5 -->

验证安装：

<details markdown="1">
<summary>查看英文原文 · 单元 5</summary>

Verify the installation:

</details>

```python
!pip list | grep transformer
!pip list | grep accelerate
```

<!-- cell:7 -->

<a id="4-provide-your-hugging-face-token"></a>

### 4. 提供您的拥抱令牌

您需要 Hugging Face API 令牌才能访问 meta-llama/Llama-3.1-8B-Instruct。在 Hugging Face Tokens 生成您的令牌并请求访问 [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)。令牌通常以“hf_”开头。

在 Jupyter Notebook中运行以下交互式块来设置令牌：

**注意**：取消选中“添加令牌作为 Git 凭证？”选项。

<details markdown="1">
<summary>查看英文原文 · 单元 7</summary>

### 4. Provide your Hugging Face token

You'll require a Hugging Face API token to access meta-llama/Llama-3.1-8B-Instruct. Generate your token at Hugging Face Tokens and request access for [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct). Tokens typically start with "hf_".

Run the following interactive block in your Jupyter notebook to set up the token:

**Note**: Uncheck the "Add token as Git credential?" option.

</details>

```python
from huggingface_hub import notebook_login, HfApi

# Prompt the user to log in
notebook_login()
```

<!-- cell:9 -->

验证您的令牌是否被正确接受：

<details markdown="1">
<summary>查看英文原文 · 单元 9</summary>

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

<!-- cell:11 -->

<a id="run-llm-inference-using-hugging-face-transformers"></a>

## 使用 Hugging Face Transformers 运行 LLM 推理

在 Docker 容器内，使用 Jupyter Notebook 运行以下代码示例：

<details markdown="1">
<summary>查看英文原文 · 单元 11</summary>

## Run LLM inference using Hugging Face Transformers

Inside the Docker container, run the following code sample using Jupyter Notebook:

</details>

```python
import transformers
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)
query = "Explain the concept of AI."
messages = [
    {"role": "system", "content": "You are an expert in the field of AI. Make sure to provide an explanation in few sentences."},
    {"role": "user", "content": query},
]

outputs = pipeline(
    messages,
    max_new_tokens=512,
    top_p = 0.7,     
    temperature=0.2,               
)

response = outputs[0]["generated_text"][-1]['content']
print('-------------------------------')
print('Query:\n', query)
print('-------------------------------')
print('Response:\n', response)
```

<!-- cell:13 -->

成功运行后，输出将如下所示：


查询：
```
 Explain the concept of AI.
```
回复：
```
 Artificial Intelligence (AI) refers to the development of computer systems that can perform tasks that typically require human intelligence, such as learning, problem-solving, decision-making, and perception. These systems use algorithms and data to simulate human-like behavior, enabling them to adapt to new situations and improve their performance over time. AI can be categorized into two main types: Narrow or Weak AI, which is designed to perform a specific task, and General or Strong AI, which aims to replicate human intelligence and reasoning across a wide range of tasks.
```

<details markdown="1">
<summary>查看英文原文 · 单元 13</summary>

After a successful run, the output will look like this:


Query:
```
 Explain the concept of AI.
```
Response:
```
 Artificial Intelligence (AI) refers to the development of computer systems that can perform tasks that typically require human intelligence, such as learning, problem-solving, decision-making, and perception. These systems use algorithms and data to simulate human-like behavior, enabling them to adapt to new situations and improve their performance over time. AI can be categorized into two main types: Narrow or Weak AI, which is designed to perform a specific task, and General or Strong AI, which aims to replicate human intelligence and reasoning across a wide range of tasks.
```

</details>
