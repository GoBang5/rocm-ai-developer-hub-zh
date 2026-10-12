> 原文作者：Mahdi Ghodsi  
> [AMD 官方教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/build_airbnb_agent_mcp.html) · [固定版本源码](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/inference/build_airbnb_agent_mcp.ipynb)  
> 上游版本：`0071bfd4bd3bd9525405c09a6ae69c983b09312a` · 同步日期：2026-10-12  
> 翻译状态：机器初译，待逐篇人工校对；代码与命令保持原文，未在 AMD GPU 上运行验证。

<!-- cell:1 -->

<a id="ai-agent-with-mcps-using-vllm-and-pydanticai"></a>

# 使用 vLLM、PydanticAI 与 MCP 构建 AI 智能体
**作者**：Mahdi Ghodsi
**知识水平**：初学者

[![在 AMD Developer Cloud 中打开](../../original/assets/jupyter_notebook_button_fnl_small.png)](https://notebooks.amd.com/tutorials/build_airbnb_agent_mcp.ipynb)

本教程利用 AMD GPU 和 **模型上下文协议 (MCP)**（一种使用 API 公开 LLM 工具的开放标准）来部署 Qwen3 等强大的语言模型。本教程的关键组件是：

- 🖥️ **vLLM** 针对 GPU 优化的 推理
- 🛠️ **PydanticAI** 用于智能体和工具管理
- 🔌 **MCP 服务器**，用于预构建工具集成

您将学习如何设置环境、部署 Qwen3 等大型语言模型、使用 MCP 将它们连接到现实世界的工具，以及构建能够推理和采取行动的对话智能体。

在本次研讨会结束时，您将构建一个由 AI 驱动的 Airbnb 助理智能体，它可以根据位置、预算和旅行日期等偏好找到住宿地点。

本教程包括以下部分：

- [1.启动 vLLM 服务器](#step1)
- [2.安装依赖项](#step2)
- [3.创建 PydanticAI 智能体的简单实例](#step3)
- [4.为您的智能体编写日期/时间工具](#step4)
- [5.将日期/时间工具替换为 MCP 服务器](#step5)
- [6.将您的智能体变成 Airbnb 房源查找助手](#step6)
- [7.挑战：扩展智能体能力](#step7)

<details markdown="1">
<summary>查看英文原文 · 单元 1</summary>

# AI agent with MCPs using vLLM and PydanticAI

**Author**: Mahdi Ghodsi  
**Knowledge level**: Beginner

[![Open in AMD Developer Cloud](../../original/assets/jupyter_notebook_button_fnl_small.png)](https://notebooks.amd.com/tutorials/build_airbnb_agent_mcp.ipynb)

This tutorial leverages AMD GPUs and **Model Context Protocol (MCP)**, an open standard for exposing LLM tools using an API, to deploy powerful language models like Qwen3. The key components for this tutorial are:

- 🖥️ **vLLM** for GPU-optimized inference
- 🛠️ **PydanticAI** for agent and tool management
- 🔌 **MCP Servers** for prebuilt tool integration

You'll learn how to set up your environment, deploy large language models like Qwen3, connect them to real-world tools using MCP, and build a conversational agent capable of reasoning and taking actions.

By the end of this workshop, you’ll have built an AI-powered Airbnb assistant agent that can find a place to stay based on preferences like location, budget, and travel dates.

This tutorial includes the following sections:

- [1. Launching the vLLM server](#step1)
- [2. Installing dependencies](#step2)
- [3. Create a simple instance of a PydanticAI agent](#step3)
- [4. Write a date/time tool for your agent](#step4)
- [5. Replace the date/time tool with an MCP server](#step5)
- [6. Turn your agent into an Airbnb finder](#step6)
- [7. Challenge to expand the agent](#step7)

</details>

<!-- cell:2 -->

<a id="prerequisites"></a>

## 前置条件

本教程是使用以下设置进行开发和测试的。 

<a id="operating-system"></a>

### 操作系统

* **Ubuntu 22.04/24.04**：确保您的系统运行的是 Ubuntu 版本 22.04 或 24.04。

<a id="hardware"></a>

### 硬件

* **AMD Instinct™ GPU**：本教程在 AMD Instinct MI300X GPU 上进行了测试。确保您使用的是 AMD Instinct GPU 或支持 ROCm 的兼容硬件，并且您的系统满足[官方要求](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)。

<a id="software"></a>

### 软件

* **ROCm 6.3 或 6.4**：按照 [ROCm 安装指南](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html) 安装并验证 ROCm。安装后，使用以下命令确认您的设置：

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

<details markdown="1">
<summary>查看英文原文 · 单元 2</summary>

## Prerequisites

This tutorial was developed and tested using the following setup. 

### Operating system

* **Ubuntu 22.04/24.04**: Ensure your system is running Ubuntu version 22.04 or 24.04.

### Hardware

* **AMD Instinct™ GPUs**: This tutorial was tested on an AMD Instinct MI300X GPU. Ensure you are using an AMD Instinct GPU or compatible hardware with ROCm support and that your system meets the [official requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

### Software

* **ROCm 6.3 or 6.4**: Install and verify ROCm by following the [ROCm install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html). After installation, confirm your setup using:

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

</details>

<!-- cell:3 -->

<a id="prepare-the-training-environment"></a>

## 准备训练环境

请按照以下步骤配置您的教程环境：

<a id="1-pull-the-docker-image"></a>

### 1. 拉取 Docker 镜像

确保您的系统满足[系统要求](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)。

拉取本教程所需的 Docker 映像：

``` bash
docker pull rocm/vllm:latest
```

<a id="2-launch-the-docker-container"></a>

### 2.启动Docker容器

启动 Docker 容器并映射必要的目录。 

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
  rocm/vllm:latest
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

Follow these steps to configure your tutorial environment:

### 1. Pull the Docker image

Ensure your system meets the [system requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

Pull the Docker image required for this tutorial:

``` bash
docker pull rocm/vllm:latest
```

### 2. Launch the Docker container

Launch the Docker container and map the necessary directories. 

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
  rocm/vllm:latest
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

<a id="step1"></a>

<a id="step-1-launching-the-vllm-server"></a>

## 步骤 1：启动 vLLM 服务器

在本次研讨会中，您将使用 [vLLM](https://github.com/vllm-project/vllm) 作为 推理 服务引擎。 vLLM 提供了许多优势，例如快速模型执行、广泛的支持模型列表以及易用性。最重要的是，它是开源的。 

<a id="deploy-the-qwen3-30b-a3b-model-with-vllm"></a>

### 使用 vLLM 部署 Qwen3-30B-A3B 模型

<details markdown="1">
<summary>查看英文原文 · 单元 4</summary>

<a id="step1"></a>

## Step 1: Launching the vLLM server

In this workshop, you'll use [vLLM](https://github.com/vllm-project/vllm) as your inference serving engine. vLLM provides many benefits, such as fast model execution, an extensive list of supported models, and ease of use. Best of all, it's open source. 

### Deploy the Qwen3-30B-A3B model with vLLM

</details>

<!-- cell:5 -->

是时候启动您的 vLLM 服务器并为您的 LLM 创建端点了。首先使用 Jupyter 服务器打开终端。然后在此终端中运行以下命令来启动 vLLM 服务器：

```bash
VLLM_USE_TRITON_FLASH_ATTN=0 \
vllm serve Qwen/Qwen3-30B-A3B \
    --served-model-name Qwen3-30B-A3B \
    --api-key abc-123 \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --trust-remote-code \
    --gpu-memory-utilization 0.9
```

打开另一个终端并通过运行此命令监视 GPU 利用率。

**注意**：对于 ROCm 6.4 及更早版本，请改用 `rocm-smi` 命令。

```bash
watch amd-smi
```

成功启动后，您的服务器应该通过 OpenAI 兼容的 API 接受传入流量。现在为您的服务器设置一些环境变量以在本教程中使用：

<details markdown="1">
<summary>查看英文原文 · 单元 5</summary>

It's time to start your vLLM server and create an endpoint for your LLM. First open a terminal using your Jupyter server. Then run the following command in this terminal to start the vLLM server:

```bash
VLLM_USE_TRITON_FLASH_ATTN=0 \
vllm serve Qwen/Qwen3-30B-A3B \
    --served-model-name Qwen3-30B-A3B \
    --api-key abc-123 \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --trust-remote-code \
    --gpu-memory-utilization 0.9
```

Open another terminal and monitor the GPU utilization by running this command.

**Note**: For ROCm 6.4 and earlier, use the `rocm-smi` command instead.

```bash
watch amd-smi
```

After a successful launch, your server should be accepting incoming traffic through an OpenAI-compatible API. Now set some environment variables for your server to use throughout this tutorial:

</details>

```python
import os

BASE_URL = f"http://localhost:8000/v1"

os.environ["BASE_URL"]    = BASE_URL
os.environ["OPENAI_API_KEY"] = "abc-123"   

print("Config set:", BASE_URL)
```

<!-- cell:7 -->

通过运行以下命令验证模型在您刚刚设置的 `BASE_URL` 中是否可用。

<details markdown="1">
<summary>查看英文原文 · 单元 7</summary>

Verify the model is available at the `BASE_URL` you just set by running the following command.

</details>

```python
!curl http://localhost:8000/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"
```

<!-- cell:9 -->

恭喜，您刚刚启动了一个功能强大的服务器，可以为任何传入请求提供服务，从而使您能够构建令人惊叹的应用程序。

<a id="step2"></a>

<a id="step-2-installing-dependencies"></a>

## 第 2 步：安装依赖项

使用以下命令安装 PydanticAI 依赖项：

<details markdown="1">
<summary>查看英文原文 · 单元 9</summary>

Congratulations, you just launched a powerful server that can serve any incoming request, allowing you to build amazing applications.

<a id="step2"></a>

## Step 2: Installing dependencies

Install the PydanticAI dependencies using this command:

</details>

```python
!pip install "pydantic-ai-slim[mcp]==1.65.0"
```

<!-- cell:11 -->

<a id="step3"></a>

<a id="step-3-create-a-simple-instance-of-a-pydanticai-agent"></a>

## 步骤 3：创建 PydanticAI 智能体的简单实例

首先为您的智能体创建一个与 OpenAI 兼容的自定义端点。

<details markdown="1">
<summary>查看英文原文 · 单元 11</summary>

<a id="step3"></a>

## Step 3: Create a simple instance of a PydanticAI agent

Start by creating a custom OpenAI-compatible endpoint for your agent.

</details>

```python
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

provider = OpenAIProvider(
    base_url=os.environ["BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
)

agent_model = OpenAIChatModel("Qwen3-30B-A3B", provider=provider)
```

<!-- cell:13 -->

现在从 `pydantic_ai` 创建 `Agent` 类的实例。

<details markdown="1">
<summary>查看英文原文 · 单元 13</summary>

Now create an instance of the `Agent` class from `pydantic_ai`.

</details>

```python
from pydantic_ai import Agent

agent = Agent(
    model=agent_model
)
```

<!-- cell:15 -->

是时候测试智能体了。 `pydantic_ai` 框架提供了多种运行 `Agent` 实例的方法。要了解更多信息，请参阅 [PydanticAI 网站](https://ai.pydantic.dev/agents/#running-agents)。

在本次研讨会中，您将在 `async` 模式下运行智能体。定义一个帮助函数，让您可以快速测试您的智能体。

<details markdown="1">
<summary>查看英文原文 · 单元 15</summary>

It's time to test the agent. The `pydantic_ai` framework provides multiple ways to run an `Agent` instance. To learn more, see the [PydanticAI site](https://ai.pydantic.dev/agents/#running-agents).

In this workshop, you are running the agent in `async` mode. Define a helper function that allows you to quickly test your agent.

</details>

```python
import asyncio
from pydantic_ai.mcp import MCPServerStdio
async def run_async(prompt: str) -> str:
    async with agent.run_mcp_servers():
        result = await agent.run(prompt)
        return result.output
```

<!-- cell:17 -->

通过调用此函数来测试智能体。

<details markdown="1">
<summary>查看英文原文 · 单元 17</summary>

Test the agent by calling this function.

</details>

```python
await run_async("What is the capital of France?")
```

<!-- cell:19 -->

现在您已经掌握了智能体实例的基础知识，请将其连接到您使用 vLLM 提供服务的模型。

<details markdown="1">
<summary>查看英文原文 · 单元 19</summary>

Now that you have the basics of an agent instance, connect it to the model you are serving with vLLM.

</details>

<!-- cell:20 -->

<a id="step4"></a>

<a id="step-4-write-a-datetime-tool-for-your-agent"></a>

## 步骤 4：为您的智能体编写日期/时间工具

法学硕士依靠他们的培训数据来响应您的提示，因此您刚刚定义的智能体将无法回答超出其培训知识范围的事实问题。您可以用一个例子来展示这一点：

<details markdown="1">
<summary>查看英文原文 · 单元 20</summary>

<a id="step4"></a>

## Step 4: Write a date/time tool for your agent

LLMs rely on their training data to respond to your prompts, so the agent you just defined would fail to answer a factual question that falls outside of its training knowledge. You can show this with an example:

</details>

```python
await run_async("What’s the date today?")
```

<!-- cell:22 -->

该模型未能回答这个问题也就不足为奇了。现在是时候通过为智能体提供可以获取当前日期的函数来启动您的 LLM 了。 LLM触发函数调用的过程通常称为“工具调用”或“函数调用”。在本次研讨会中，您将利用 `pydantic_ai` 智能体程序包 `Tool` 为智能体程序提供适当的工具。首先，使用下面的代码示例在此框架中定义自定义工具。

<details markdown="1">
<summary>查看英文原文 · 单元 22</summary>

It's no surprise that the model failed to answer this question. Now it's time to power-up your LLM by providing the agent with a function that can get the current date. The process by which an LLM triggers a function call is commonly referred to as "Tool Calling" or "Function Calling". In this workshop, you're going to take advantage of the `pydantic_ai` agent `Tool` package to provide the agent with appropriate tools. First, define a custom tool within this framework using the code sample below.

</details>

```python
from datetime import datetime
from pydantic_ai import Tool          
@Tool
def get_current_date() -> str:
    """Return the current date/time as an ISO-formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

<!-- cell:24 -->

接下来，通过将该工具的函数签名添加到 `Agent` 构造函数来将该工具提供给您的智能体。这会通知 LLM 新工具存在。

<details markdown="1">
<summary>查看英文原文 · 单元 24</summary>

Next, provide this tool to your agent by adding the function signature of the tool to the `Agent` constructor. This notifies the LLM that the new tool exists.

</details>

```python
agent = Agent(
    model=agent_model,
    tools=[get_current_date],
    system_prompt = (
        "You have access to:\n"
        "   1. get_current_time(params: dict)\n"
        "Use this tool for date/time questions."
    )
)
```

<!-- cell:26 -->

现在测试智能体。

<details markdown="1">
<summary>查看英文原文 · 单元 26</summary>

Now test the agent.

</details>

```python
await run_async("What’s the date today?")
```

<!-- cell:28 -->

恭喜您构建了一个可以访问实时数据的智能体。

<details markdown="1">
<summary>查看英文原文 · 单元 28</summary>

Congratulations on building an agent with access to real-time data.

</details>

<!-- cell:29 -->

<a id="step5"></a>

<a id="step-5-replace-the-datetime-tool-with-an-mcp-server"></a>

## 步骤 5：用 MCP 服务器替换日期/时间工具

现在您已了解如何创建自定义工具并让智能体访问它，您可以使用 [模型上下文协议](https://modelcontextprotocol.io/introduction) (MCP) 增强此步骤。您可以将自定义工具替换为为智能体提供服务并提供类似信息的简单 MCP 服务器。

为什么应该使用 MCP？ MCP 服务器提供：
- ✅ 标准化API接口
- 🔄 跨项目的可重用性
- 📦 预建功能

要将自定义时间工具替换为官方 MCP 时间服务器，请按照以下步骤操作：

<a id="installing-an-mcp-time-server"></a>

### 安装 MCP 时间服务器

首先安装 MCP 服务器：

<details markdown="1">
<summary>查看英文原文 · 单元 29</summary>

<a id="step5"></a>

## Step 5: Replace the date/time tool with an MCP server

Now that you learned how to create a custom tool and let the agent access it, you can enhance this step using the [Model Context Protocol](https://modelcontextprotocol.io/introduction) (MCP). You can replace the custom tool with a simple MCP server that serves the agent and provides similar information.

Why should you use MCP? MCP servers provide:
- ✅ Standardized API interfaces
- 🔄 Reusability across projects
- 📦 Prebuilt functionality

To replace your custom time tool with an official MCP time server, follow these steps:

### Installing an MCP time server

Start by installing the MCP server:

</details>

```python
!pip install -q mcp-server-time
```

<!-- cell:31 -->

现在定义 `time_server`：

<details markdown="1">
<summary>查看英文原文 · 单元 31</summary>

Now define the `time_server`:

</details>

```python
from pydantic_ai.mcp import MCPServerStdio

time_server = MCPServerStdio(
    "python",
    args=[
        "-m", "mcp_server_time",
        "--local-timezone=America/New_York",
    ],
)
```

<!-- cell:33 -->

最后，通过删除先前定义的工具并添加 MCP 服务器来修改您的智能体。

<details markdown="1">
<summary>查看英文原文 · 单元 33</summary>

Finally, modify your agent by removing the previously defined tool and adding the MCP server instead.

</details>

```python
agent = Agent(
    model=agent_model,
    toolsets=[time_server],
    system_prompt = (
        "You are a helpful agent and you have access to this tool:\n"
        "   get_current_time(params: dict)\n"
        "When the user asks for the current date or time, call get_current_time.\n"
    )
)
```

<!-- cell:35 -->

您现在可以查看智能体是否可以使用 MCP 来提供正确的时间。

<details markdown="1">
<summary>查看英文原文 · 单元 35</summary>

You can now see whether the agent can use MCP to provide the correct time.

</details>

```python
await run_async("What’s the date today?")
```

<!-- cell:37 -->

您现在已经使用官方 MCP 服务器来启动您的智能体。在下一部分中，您将学习如何使用数百个可用的免费或付费 MCP 服务器将您的想法转化为实际的工作项目。


<a id="step6"></a>

<a id="step-6-turn-your-agent-into-an-airbnb-finder"></a>

## 第 6 步：将您的智能体人转变为 Airbnb 房源查找助手

正如您在上一节中发现的，MCP 服务器非常易于使用。它们提供了一种为法学硕士提供所需工具的标准方法。已有数千台 MCP 服务器可供您使用。请参阅以下 MCP 跟踪器之一作为参考，以了解可用的服务器：
- [https://github.com/modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers)
- [https://mcp.so/](https://mcp.so/)

您将使用 npx 启动下一个服务器。使用以下命令安装所需的依赖项：

<details markdown="1">
<summary>查看英文原文 · 单元 37</summary>

You have now used an official MCP server to power up your agent. In the next section, you'll learn how to turn your ideas into real working projects by using the hundreds of available free or paid MCP servers.


<a id="step6"></a>

## Step 6: Turn your agent into an Airbnb finder

As you discovered in the last section, MCP servers are very easy to use. They provide a standard way of providing LLMs with the tools they need. There are already thousands of MCP servers available for you to use. Consult one of the following MCP trackers as a reference to find out about the available servers:
- [https://github.com/modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers)
- [https://mcp.so/](https://mcp.so/)

You will use npx to launch your next server. Use the following commands to install the required dependencies:

</details>

```python
# Install Node.js 20 via NodeSource
!curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
!apt install -y nodejs
```

<!-- cell:39 -->

验证 `npm` 和 `npx` 安装：

<details markdown="1">
<summary>查看英文原文 · 单元 39</summary>

Verify the `npm` and `npx` installations:

</details>

```python
!node -v && npm -v && npx --version
```

<!-- cell:41 -->

在研讨会的这一部分中，您将构建一个智能体来帮助浏览可供预订的 Airbnb 房源。您可以在已完成的基础上进行构建，并将开源 Airbnb MCP 服务器添加到您的智能体中。首先定义 Airbnb 服务器。

<details markdown="1">
<summary>查看英文原文 · 单元 41</summary>

In this part of the workshop, you're going to build an agent to help browse available Airbnbs listings to book. You can build on top of what you've done already and add an open-source Airbnb MCP server to your agent. Start by defining the Airbnb server.

</details>

```python
airbnb_server = MCPServerStdio(
    "npx", args=["-y", "@openbnb/mcp-server-airbnb", "--ignore-robots-txt"]
)
```

<!-- cell:43 -->

现在更新您的智能体。

<details markdown="1">
<summary>查看英文原文 · 单元 43</summary>

Now update your agent.

</details>

```python
system_prompt = """
You have access to three tools:
1. get_current_time(params: dict)
2. airbnb_search(params: dict)
3. airbnb_listing_details(params: dict)
When the user asks for listings, first call get_current_time, then airbnb_search, etc.
"""

agent = Agent(
    model=agent_model,
    toolsets=[time_server, airbnb_server],
    system_prompt=system_prompt,
)
```

<!-- cell:45 -->

最后，看看您的智能体是否可以浏览 Airbnb 房源。

<details markdown="1">
<summary>查看英文原文 · 单元 45</summary>

Finally, see if your agent can browse through the Airbnb listings.

</details>

```python
await run_async("Find a place to stay in Vancouver for next Sunday for 3 nights for 2 adults?")
```

<!-- cell:47 -->

<a id="step7"></a>

<a id="step-7-challenge-to-expand-the-agent"></a>

## 第 7 步：挑战：扩展智能体能力

对于额外的挑战，请使用您选择的 MCP 天气服务器添加天气集成：
1.启动MCP天气服务器
2.将其添加到智能体工具列表中
3.请智能体根据天气建议最佳旅行日期

<details markdown="1">
<summary>查看英文原文 · 单元 47</summary>

<a id="step7"></a>

## Step 7: Challenge to expand the agent

For an additional challenge, add weather integration using the MCP weather server of your choice:
1. Launch the MCP weather server
2. Add it to the list of agent tools
3. Ask the agent to suggest the best travel dates based on the weather

</details>
