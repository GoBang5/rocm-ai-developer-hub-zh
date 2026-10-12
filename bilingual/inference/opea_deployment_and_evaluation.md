> 原文作者：Yu Wang  
> [AMD 官方教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/opea_deployment_and_evaluation.html) · [固定版本源码](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/inference/opea_deployment_and_evaluation.ipynb)  
> 上游版本：`0071bfd4bd3bd9525405c09a6ae69c983b09312a` · 同步日期：2026-10-12  
> 翻译状态：机器初译，待逐篇人工校对；代码与命令保持原文，未在 AMD GPU 上运行验证。

<!-- cell:1 -->

<a id="chatqna-vllm-deployment-and-performance-evaluation"></a>

# ChatQnA：vLLM 部署与性能评估
**作者**：Yu Wang
**知识水平**：初学者

<details markdown="1">
<summary>查看英文原文 · 单元 1</summary>

# ChatQnA vLLM deployment and performance evaluation

**Author**: Yu Wang  
**Knowledge level**: Beginner

</details>

<!-- cell:2 -->

ChatQnA 是一个 检索增强生成 (RAG) 系统，它将文档检索与 LLM 推理 相结合。本教程提供了在支持 ROCm 的 AMD GPU 上使用 vLLM 部署 ChatQnA 以及评估管道性能的综合指南。

<a id="key-features"></a>

## 主要特点

以下是使用 ChatQnA 的好处：

- **vLLM 集成**：LLM 在 AMD Instinct™ GPU 上提供优化的 推理
- **AMD GPU 支持**：基于 ROCm 的 GPU 加速
- **矢量搜索**：基于Redis的文档检索
- **RAG管道**：完整的问答系统
- **性能监控**：内置指标和评估工具

<a id="overview"></a>

## 概述

本教程包括以下部分：

1. [前置条件](#prerequisites)
2. [准备环境](#prepare-the-environment)
3. [系统架构](#system-architecture)
4. [部署指南](#deployment-guide)
5. [性能评估](#performance-evaluation)
6. [常见问题及解决方法](#common-issues-and-solutions)
7. [高级配置](#advanced-configuration)
8. [故障排除](#troubleshooting)

<details markdown="1">
<summary>查看英文原文 · 单元 2</summary>

ChatQnA is a Retrieval-Augmented Generation (RAG) system that combines document retrieval with LLM inference. This tutorial provides a comprehensive guide for deploying ChatQnA using vLLM on AMD GPUs with ROCm support, as well as evaluating pipeline performance.

## Key features

Here are the benefits of using ChatQnA:

- **vLLM integration**: LLM serving with optimized inference on AMD Instinct™ GPUs
- **AMD GPU support**: ROCm-based GPU acceleration
- **Vector search**: Redis-based document retrieval
- **RAG pipeline**: Complete question-answering system
- **Performance monitoring**: Built-in metrics and evaluation tools

## Overview

This tutorial includes the following sections:

1. [Prerequisites](#prerequisites)
2. [Prepare the environment](#prepare-the-environment)
3. [System architecture](#system-architecture)
4. [Deployment guide](#deployment-guide)
5. [Performance evaluation](#performance-evaluation)
6. [Common issues and solutions](#common-issues-and-solutions)
7. [Advanced configuration](#advanced-configuration)
8. [Troubleshooting](#troubleshooting)

</details>

<!-- cell:3 -->

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

* **ROCm 6.3 或 6.4**：按照 [ROCm 安装指南](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html) 安装并验证 ROCm。 

<a id="hugging-face-api-access"></a>

### Hugging Face API 访问

* 从[Hugging Face](https://huggingface.co)获取API令牌。

<a id="prepare-the-environment"></a>

## 准备环境

本节创建虚拟环境，然后启动 Jupyter 服务器。 

<a id="set-up-a-virtual-environment"></a>

### 设置虚拟环境

首先创建一个虚拟环境：

``` bash
python3 -m venv venv
source venv/bin/activate 
```

<a id="install-and-launch-jupyter"></a>

### 安装并启动 Jupyter

使用以下命令安装 Jupyter：

```bash
pip install jupyter
```

然后启动Jupyter服务器：

```bash
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**注意**：在运行上述命令之前，请确保您的系统上尚未使用端口 `8888`。如果是，您可以通过将 `--port=8888` 替换为其他端口号来指定其他端口，例如 `--port=8890`。

<details markdown="1">
<summary>查看英文原文 · 单元 3</summary>

## Prerequisites

This tutorial was developed and tested using the following setup. 

### Operating system

* **Ubuntu 22.04/24.04**: Ensure your system is running Ubuntu version 22.04 or 24.04.

### Hardware

* **AMD Instinct™ GPUs**: This tutorial was tested on an AMD Instinct MI300X GPU. Ensure you are using an AMD Instinct GPU or compatible hardware with ROCm support and that your system meets the [official requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

### Software

* **ROCm 6.3 or 6.4**: Install and verify ROCm by following the [ROCm install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html). 

### Hugging Face API access

* Obtain an API token from [Hugging Face](https://huggingface.co).

## Prepare the environment

This section creates a virtual environment and then starts the Jupyter server. 

### Set up a virtual environment

Start by creating a virtual environment:

``` bash
python3 -m venv venv
source venv/bin/activate 
```

### Install and launch Jupyter

Install Jupyter using the following command:

```bash
pip install jupyter
```

Then start the Jupyter server:

```bash
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**Note**: Ensure port `8888` is not already in use on your system before running the above command. If it is, you can specify a different port by replacing `--port=8888` with another port number, for example, `--port=8890`.

</details>

<!-- cell:4 -->

<a id="system-architecture"></a>

## 系统架构

本节介绍 ChatQnA 架构，包括服务和数据流。

<a id="service-components"></a>

### 服务组件

下图展示了完整的ChatQnA系统架构。

**架构概述：**
```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                               EXTERNAL ACCESS                                     │
│                                                                                   │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐   │
│   │   Web browser   │    │   API clients   │    │      Monitoring tools       │   │
│   │                 │    │                 │    │    (Grafana, Prometheus)    │   │
│   └─────────────────┘    └─────────────────┘    └─────────────────────────────┘   │
│           │                       │                           │                   │
│           │                       │                           │                   │
│           ▼                       ▼                           ▼                   │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐   │
│   │   Nginx Proxy   │    │   Backend API   │    │        Redis Insight        │   │
│   │   (Port 8081)   │    │   (Port 8890)   │    │         (Port 8002)         │   │
│   └─────────────────┘    └─────────────────┘    └─────────────────────────────┘   │
│           │                       │                           │                   │
│           │                       │                           │                   │
│           ▼                       ▼                           ▼                   │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐   │
│   │   Frontend UI   │    │     Backend     │    │   Redis Vector database     │   │
│   │   (Port 5174)   │    │     server      │    │         (Port 6380)         │   │
│   │   (React App)   │    │    (FastAPI)    │    │      (Vector storage)       │   │
│   └─────────────────┘    └─────────────────┘    └─────────────────────────────┘   │
│                                   │                           │                   │
│                                   │                           │                   │
│                                   ▼                           ▼                   │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │                             RAG PIPELINE                                    │  │
│  │                                                                             │  │
│  │  ┌───────────────────┐ ┌─────────────────────┐ ┌─────────────────────────┐  │  │
│  │  │ Retriever service │ │TEI embedding service│ │  TEI reranking service  │  │  │
│  │  │                   │ │                     │ │                         │  │  │
│  │  │   (Port 7001)     │ │    (Port 18091)     │ │      (Port 18809)       │  │  │
│  │  │                   │ │                     │ │                         │  │  │
│  │  │ • Vector search   │ │ • Text embedding    │ │ • Document reranking    │  │  │
│  │  │ • Similarity      │ │ • BGE model         │ │ • Relevance scoring     │  │  │
│  │  │   matching        │ │ • CPU inference     │ │ • CPU inference         │  │  │
│  │  └───────────────────┘ └─────────────────────┘ └─────────────────────────┘  │  │
│  │            │                      │                         │               │  │
│  │            │                      │                         │               │  │
│  │            ▼                      ▼                         ▼               │  │
│  │  ┌───────────────────────────────────────────────────────────────────────┐  │  │
│  │  │                           vLLM service                                │  │  │
│  │  │                           (Port 18009)                                │  │  │
│  │  │                                                                       │  │  │
│  │  │                  • High-performance LLM inference                     │  │  │
│  │  │                  • AMD GPU acceleration (ROCm)                        │  │  │
│  │  │                  • Qwen2.5-7B-Instruct model                          │  │  │
│  │  │                  • Optimized for throughput and latency               │  │  │
│  │  │                  • Tensor parallel support                            │  │  │
│  │  └───────────────────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                            │
│                                      │                                            │
│                                      ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │                            DATA PIPELINE                                    │  │
│  │                                                                             │  │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │  │
│  │  │   Dataprep      │    │   Model cache   │    │   Document storage      │  │  │
│  │  │   service       │    │   (./data)      │    │   (Redis Vector DB)     │  │  │
│  │  │   (Port 18104)  │    │                 │    │                         │  │  │
│  │  │                 │    │ • Downloaded    │    │ • Vector embeddings     │  │  │
│  │  │ • Document      │    │   models        │    │ • Metadata index        │  │  │
│  │  │   processing    │    │ • Model weights │    │ • Full-text search      │  │  │
│  │  │ • Text          │    │ • Cache storage │    │ • Similarity search     │  │  │
│  │  │   extraction    │    │ • Shared volume │    │ • Redis stack           │  │  │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────────┘
```

**附加服务：**
- **数据准备服务**（端口 `18104`）：文档处理和摄取
- **Redis Insight**（端口`8002`）：数据库监控接口
- **模型缓存** (`./data`)：模型存储的共享卷

<a id="data-flow"></a>

### 数据流

新查询的管道遵循以下步骤：

1. **用户输入**：使用前端提交问题。
2. **嵌入**：使用 TEI 服务将问题转换为向量。
3. **检索**：从Redis向量数据库中检索相似文档。
4. **重新排名**：对检索到的文档进行相关性重新排名。
5. **LLM 推理**：vLLM 使用检索到的上下文生成答案。
6. **响应**：答案通过前端返回给用户。

<details markdown="1">
<summary>查看英文原文 · 单元 4</summary>

## System architecture

This section describes the ChatQnA architecture, including the services and data flow.

### Service components

The following diagram shows the complete ChatQnA system architecture.

**Architecture Overview:**
```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                               EXTERNAL ACCESS                                     │
│                                                                                   │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐   │
│   │   Web browser   │    │   API clients   │    │      Monitoring tools       │   │
│   │                 │    │                 │    │    (Grafana, Prometheus)    │   │
│   └─────────────────┘    └─────────────────┘    └─────────────────────────────┘   │
│           │                       │                           │                   │
│           │                       │                           │                   │
│           ▼                       ▼                           ▼                   │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐   │
│   │   Nginx Proxy   │    │   Backend API   │    │        Redis Insight        │   │
│   │   (Port 8081)   │    │   (Port 8890)   │    │         (Port 8002)         │   │
│   └─────────────────┘    └─────────────────┘    └─────────────────────────────┘   │
│           │                       │                           │                   │
│           │                       │                           │                   │
│           ▼                       ▼                           ▼                   │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐   │
│   │   Frontend UI   │    │     Backend     │    │   Redis Vector database     │   │
│   │   (Port 5174)   │    │     server      │    │         (Port 6380)         │   │
│   │   (React App)   │    │    (FastAPI)    │    │      (Vector storage)       │   │
│   └─────────────────┘    └─────────────────┘    └─────────────────────────────┘   │
│                                   │                           │                   │
│                                   │                           │                   │
│                                   ▼                           ▼                   │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │                             RAG PIPELINE                                    │  │
│  │                                                                             │  │
│  │  ┌───────────────────┐ ┌─────────────────────┐ ┌─────────────────────────┐  │  │
│  │  │ Retriever service │ │TEI embedding service│ │  TEI reranking service  │  │  │
│  │  │                   │ │                     │ │                         │  │  │
│  │  │   (Port 7001)     │ │    (Port 18091)     │ │      (Port 18809)       │  │  │
│  │  │                   │ │                     │ │                         │  │  │
│  │  │ • Vector search   │ │ • Text embedding    │ │ • Document reranking    │  │  │
│  │  │ • Similarity      │ │ • BGE model         │ │ • Relevance scoring     │  │  │
│  │  │   matching        │ │ • CPU inference     │ │ • CPU inference         │  │  │
│  │  └───────────────────┘ └─────────────────────┘ └─────────────────────────┘  │  │
│  │            │                      │                         │               │  │
│  │            │                      │                         │               │  │
│  │            ▼                      ▼                         ▼               │  │
│  │  ┌───────────────────────────────────────────────────────────────────────┐  │  │
│  │  │                           vLLM service                                │  │  │
│  │  │                           (Port 18009)                                │  │  │
│  │  │                                                                       │  │  │
│  │  │                  • High-performance LLM inference                     │  │  │
│  │  │                  • AMD GPU acceleration (ROCm)                        │  │  │
│  │  │                  • Qwen2.5-7B-Instruct model                          │  │  │
│  │  │                  • Optimized for throughput and latency               │  │  │
│  │  │                  • Tensor parallel support                            │  │  │
│  │  └───────────────────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                            │
│                                      │                                            │
│                                      ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │                            DATA PIPELINE                                    │  │
│  │                                                                             │  │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │  │
│  │  │   Dataprep      │    │   Model cache   │    │   Document storage      │  │  │
│  │  │   service       │    │   (./data)      │    │   (Redis Vector DB)     │  │  │
│  │  │   (Port 18104)  │    │                 │    │                         │  │  │
│  │  │                 │    │ • Downloaded    │    │ • Vector embeddings     │  │  │
│  │  │ • Document      │    │   models        │    │ • Metadata index        │  │  │
│  │  │   processing    │    │ • Model weights │    │ • Full-text search      │  │  │
│  │  │ • Text          │    │ • Cache storage │    │ • Similarity search     │  │  │
│  │  │   extraction    │    │ • Shared volume │    │ • Redis stack           │  │  │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────────┘
```

**Additional services:**
- **Dataprep service** (Port `18104`): Document processing and ingestion
- **Redis Insight** (Port `8002`): Database monitoring interface
- **Model cache** (`./data`): Shared volume for model storage

### Data flow

The pipeline for a new query follows these steps:

1. **User input**: A question is submitted using the frontend.
2. **Embedding**: The question is converted to a vector using the TEI service.
3. **Retrieval**: Similar documents are retrieved from the Redis vector database.
4. **Reranking**: The retrieved documents are reranked for relevance.
5. **LLM inference**: vLLM generates an answer using the retrieved context.
6. **Response**: The answer is returned to the user through the frontend.

</details>

<!-- cell:5 -->

<a id="deployment-guide"></a>

## 部署指南

要部署 ChatQnA，请按照下列步骤操作：

<a id="step-1-pull-the-source-code-from-github"></a>

### 第 1 步：从 GitHub 获取源代码

首先，克隆企业 AI 开放平台 (OPEA) GenAIExamples 存储库，其中包含 ChatQnA 实现和部署所需的其他 AI 示例。

<details markdown="1">
<summary>查看英文原文 · 单元 5</summary>

## Deployment guide

To deploy ChatQnA, follow these steps:

### Step 1: Pull the source code from GitHub

First, clone the Open Platform for Enterprise AI (OPEA) GenAIExamples repository, which contains the ChatQnA implementation and other AI examples needed for your deployment.

</details>

```python
# Home directory
import os
HOME_DIR = os.getcwd()
# Open Platform for Enterprise AI (OPEA)
!git clone https://github.com/opea-project/GenAIExamples.git
```

<!-- cell:7 -->

接下来，克隆 LaunchPad 存储库，该存储库提供专为 AMD GPU 环境上的 ChatQnA 使用案例设计的一键部署脚本和配置文件。

<details markdown="1">
<summary>查看英文原文 · 单元 7</summary>

Next, clone the LaunchPad repository that provides one-click deployment scripts and configuration files specifically designed for ChatQnA use cases on AMD GPU environments.

</details>

```python
# One click deployment scripts for the use case
!git clone https://github.com/Yu-amd/LaunchPad.git
```

<!-- cell:9 -->

最后，克隆 GenAIEval 评估存储库，其中包含您将用于评估 ChatQnA 系统性能的基准测试工具。

<details markdown="1">
<summary>查看英文原文 · 单元 9</summary>

Finelly, clone the GenAIEval evaluation repository, which contains the benchmarking tools you'll use to evaluate the ChatQnA system performance.

</details>

```python
# Pipeline performance evaluation harness
!git clone https://github.com/opea-project/GenAIEval.git
```

<!-- cell:11 -->

LaunchPad 项目使用与 OPEA 项目相同的层次结构。将每个目录中的 LaunchPad 脚本和 YAML 文件复制到 OPEA 文件夹中的相应目录。

<details markdown="1">
<summary>查看英文原文 · 单元 11</summary>

The LaunchPad project uses the same hierarchy as the OPEA project. Copy the LaunchPad scripts and YAML files from each directory to the corresponding directory in the OPEA folder.

</details>

```python
# Copy necessary scripts and configuration files to the OPEA directory
!cp {HOME_DIR}/LaunchPad/GenAIExamples/ChatQnA/docker_compose/amd/gpu/rocm/*.sh {HOME_DIR}/GenAIExamples/ChatQnA/docker_compose/amd/gpu/rocm/
!cp {HOME_DIR}/LaunchPad/GenAIExamples/ChatQnA/docker_compose/amd/gpu/rocm/*.yaml {HOME_DIR}/GenAIExamples/ChatQnA/docker_compose/amd/gpu/rocm/
!cp {HOME_DIR}/LaunchPad/GenAIExamples/ChatQnA/docker_compose/amd/gpu/rocm/.env {HOME_DIR}/GenAIExamples/ChatQnA/docker_compose/amd/gpu/rocm/
!cp -r {HOME_DIR}/LaunchPad/GenAIEval/evals/benchmarks/* {HOME_DIR}/GenAIEval/evals/benchmark/
```

<!-- cell:13 -->

<a id="step-2-environment-setup"></a>

### 第 2 步：环境设置

现在导航到所有配置文件和脚本所在的 OPEA 部署目录。

<details markdown="1">
<summary>查看英文原文 · 单元 13</summary>

### Step 2: Environment setup

Now navigate to the OPEA deployment directory where all the configuration files and scripts are located.

</details>

```python
# Navigate to the OPEA deployment directory
%cd {HOME_DIR}/GenAIExamples/ChatQnA/docker_compose/amd/gpu/rocm
```

<!-- cell:15 -->

您需要配置环境变量管理。首先，安装 `python-dotenv` 软件包，它允许您从 `.env` 文件加载环境变量。然后导入必要的模块并从包含配置设置的文件中加载环境变量。

您还需要配置 Hugging Face API 令牌，这是下载 ChatQnA 系统使用的 AI 模型所必需的。将 `your_token_here` 替换为实际的 Hugging Face 令牌。

<details markdown="1">
<summary>查看英文原文 · 单元 15</summary>

You'll need to configure environment variable management. First, install the `python-dotenv` package which lets you load environment variables from an `.env` file. Then import the necessary modules and load the environment variables from the file that contains your configuration settings.

You also need to configure your Hugging Face API token, which is required to download the AI models used by the ChatQnA system. Replace `your_token_here` with the actual Hugging Face token.

</details>

```python
# Install and load .env
!pip install python-dotenv
# Configure Hugging Face token
!sed -i 's/CHATQNA_HUGGINGFACEHUB_API_TOKEN=""/CHATQNA_HUGGINGFACEHUB_API_TOKEN="YOUR_ACTUAL_TOKEN_HERE"/' .env
```

<!-- cell:17 -->

设置所有环境变量。

<details markdown="1">
<summary>查看英文原文 · 单元 17</summary>

Set all the environment variables.

</details>

```python
# Load environment variables
from dotenv import load_dotenv
import os
load_dotenv()  # Loads variables from .env file
```

<!-- cell:19 -->

现在使用提供的脚本设置 vLLM 环境。这会在 AMD GPU 上配置高性能 LLM 推理 所需的所有组件。

<details markdown="1">
<summary>查看英文原文 · 单元 19</summary>

Now set up the vLLM environment using the provided script. This configures all the necessary components for high-performance LLM inference on AMD GPUs.

</details>

```python
# Setup vLLM environment
!./run_chatqna.sh setup-vllm
```

<!-- cell:21 -->

<a id="step-3-deploy-the-workload"></a>

### 步骤 3：部署工作负载

配置好环境后，您现在可以启动 vLLM 服务。这将启动 ChatQnA 系统所需的所有容器和服务。

<details markdown="1">
<summary>查看英文原文 · 单元 21</summary>

### Step 3: Deploy the workload

With the environment configured, you can now start the vLLM services. This launches all the necessary containers and services for the ChatQnA system.

</details>

```python
# Start vLLM services
import subprocess
subprocess.run(["./run_chatqna.sh", "start-vllm"], check=True)
```

<!-- cell:23 -->

检查所有正在运行的服务的状态，以确保它们正确启动并正常运行。

<details markdown="1">
<summary>查看英文原文 · 单元 23</summary>

Check the status of all the running services to ensure they started correctly and are functioning properly.

</details>

```python
# Check service status
!./run_chatqna.sh status
```

<!-- cell:25 -->

监视 vLLM 服务日志 60 秒以验证服务是否正确启动。查看日志中是否有任何初始化问题。

<details markdown="1">
<summary>查看英文原文 · 单元 25</summary>

Monitor the vLLM service logs for 60 seconds to verify the service starts correctly. Review the logs for any initialization issues.

</details>

```python
# Check chatqna-vllm-service status
!timeout 200 docker logs -f chatqna-vllm-service
```

<!-- cell:27 -->

<a id="step-4-verify-the-deployment"></a>

### 步骤 4：验证部署

通过检查其状态和端口映射来验证所有 Docker 容器是否正常运行。

<details markdown="1">
<summary>查看英文原文 · 单元 27</summary>

### Step 4: Verify the deployment

Verify that all Docker containers are running properly by checking their status and port mappings.

</details>

```python
# Check running containers
!docker ps
```

<!-- cell:29 -->

下一个命令向后端 API 发送一条简单的测试消息，以验证 ChatQnA 服务是否正常工作。

<details markdown="1">
<summary>查看英文原文 · 单元 29</summary>

The next command sends a simple test message to the backend API to verify the ChatQnA service is working properly.

</details>

```python
# Test backend API
!curl -X POST http://localhost:8890/v1/chatqna \
  -H "Content-Type: application/json" \
  -d '{"messages": "Hello, how are you?"}'
```

<!-- cell:31 -->

<a id="step-5-upload-documents"></a>

### 第5步：上传文件

创建示例文档并将其上传到系统。这演示了如何将文档输入 ChatQnA 系统以进行检索和问答。

<details markdown="1">
<summary>查看英文原文 · 单元 31</summary>

### Step 5: Upload documents

Create a sample document and upload it to the system. This demonstrates how to feed documents into the ChatQnA system for retrieval and question answering.

</details>

```python
# Create a text file
!echo "Your document content here" > document.txt

# Upload the file
!curl -X POST http://localhost:18104/v1/dataprep/ingest \
  -H "Content-Type: multipart/form-data" \
  -F "files=@document.txt"
```

<!-- cell:33 -->

现在通过检查 Redis 矢量数据库的内容来验证文档是否已成功上传并建立索引。

<details markdown="1">
<summary>查看英文原文 · 单元 33</summary>

Now verify that the document was successfully uploaded and indexed by checking the contents of the Redis vector database.

</details>

```python
# Verify the upload worked
# Check if the document was indexed
!curl -X POST http://localhost:18104/v1/dataprep/get \
  -H "Content-Type: application/json" \
  -d '{"index_name": "rag-redis"}'
```

<!-- cell:35 -->

您还可以一次上传多个文档。以下介绍了如何同时创建和上传多个文档以建立您的知识库。

<details markdown="1">
<summary>查看英文原文 · 单元 35</summary>

You can also upload multiple documents at once. Here's how to create and upload several documents simultaneously to build up your knowledge base.

</details>

```python
# For multiple documents
# Create multiple files
!echo "Document 1 content" > doc1.txt
!echo "Document 2 content" > doc2.txt

# Upload multiple files
!curl -X POST http://localhost:18104/v1/dataprep/ingest \
  -H "Content-Type: multipart/form-data" \
  -F "files=@doc1.txt" \
  -F "files=@doc2.txt"
```

<!-- cell:37 -->

<a id="performance-evaluation"></a>

## 绩效评估

绩效评估可帮助您了解以下指标：

- **吞吐量**：每秒请求数
- **延迟**：响应时间
- **准确性**：答案质量
- **资源使用**：CPU、GPU 和内存利用率

<a id="step-1-set-up-the-evaluation-environment"></a>

### 第 1 步：设置评估环境

导航到 GenAIEval 目录以设置并运行性能评估测试。

<details markdown="1">
<summary>查看英文原文 · 单元 37</summary>

## Performance evaluation

Performance evaluation helps you understand the following metrics:

- **Throughput**: Requests per second
- **Latency**: Response time
- **Accuracy**: Answer quality
- **Resource usage**: CPU, GPU, and memory utilization

### Step 1: Set up the evaluation environment

Navigate to the GenAIEval directory to set up and run your performance evaluation tests.

</details>

```python
# Navigate to evaluation directory
%cd {HOME_DIR}/GenAIEval
```

<!-- cell:39 -->

安装评估工具所需的依赖项并在开发模式下设置 GenAIEval 包。

<details markdown="1">
<summary>查看英文原文 · 单元 39</summary>

Install the required dependencies for the evaluation tools and set up the GenAIEval package in development mode.

</details>

```python
# Install evaluation dependencies
!pip install -r requirements.txt
!pip install -e .
```

<!-- cell:41 -->

<a id="step-2-run-the-basic-evaluation"></a>

### 第 2 步：运行基本评估

现在导航回 ChatQnA 部署目录并在已部署的系统上运行性能评估测试。

<details markdown="1">
<summary>查看英文原文 · 单元 41</summary>

### Step 2: Run the basic evaluation

Now navigate back to the ChatQnA deployment directory and run the performance evaluation tests on your deployed system.

</details>

```python
# Navigate back to GenAIExamples/ChatQnA/docker_compose/amd/gpu/rocm/
%cd {HOME_DIR}/GenAIExamples/ChatQnA/docker_compose/amd/gpu/rocm/
```

<!-- cell:43 -->

现在运行 vLLM 评估脚本，该脚本测试 ChatQnA 系统的性能，测量 吞吐量、延迟 等指标和响应质量。

<details markdown="1">
<summary>查看英文原文 · 单元 43</summary>

Now run the vLLM evaluation script, which tests the performance of your ChatQnA system, measuring metrics like throughput, latency, and response quality.

</details>

```python
# Run vLLM evaluation
!./run_chatqna.sh vllm-eval
```

<!-- cell:45 -->

<a id="step-3-performance-metrics"></a>

### 步骤 3：绩效指标

本部分执行额外的 吞吐量 和 延迟 测试。

<a id="throughput-testing"></a>

#### 吞吐量 测试

安装 Apache Bench (ab)，它执行负载测试并测量各种条件下 ChatQnA API 的 吞吐量。

<details markdown="1">
<summary>查看英文原文 · 单元 45</summary>

### Step 3: Performance metrics

This section performs additional throughput and latency testing.

#### Throughput testing

Install Apache Bench (ab), which performs load testing and measures the throughput of the ChatQnA API under various conditions.

</details>

```python
# Install dependency
!apt install -y apache2-utils
```

<!-- cell:47 -->

创建包含复杂问题的测试文件，以评估系统处理详细的多部分查询并生成全面响应的能力。

<details markdown="1">
<summary>查看英文原文 · 单元 47</summary>

Create a test file with a complex question to evaluate how well the system handles detailed, multi-part queries and generates comprehensive responses.

</details>

```python
# Create a complex test file
!echo '{"messages": "Can you provide a detailed explanation of how neural networks work, including the concepts of forward propagation, backpropagation, and gradient descent? Also explain how these concepts relate to deep learning and why they are important for modern AI systems."}' > test_data.json
```

<!-- cell:49 -->

现在使用 Apache Bench 运行负载测试，模拟 100 个并发请求和 10 个并发连接。这可以衡量系统的 吞吐量 和压力下的性能。

<details markdown="1">
<summary>查看英文原文 · 单元 49</summary>

Now use Apache Bench to run a load test simulating 100 concurrent requests with 10 simultaneous connections. This measures the system's throughput and performance under stress.

</details>

```python
# Test concurrent requests
!ab -n 100 -c 10 -p test_data.json -T application/json \
  http://localhost:8890/v1/chatqna
```

<!-- cell:51 -->

<a id="latency-testing"></a>

#### 延迟 测试

为 `curl` 创建详细的计时格式文件，以帮助测量各种 延迟 指标，包括 DNS 查找、连接时间和总响应时间，以进行精确的性能分析。

<details markdown="1">
<summary>查看英文原文 · 单元 51</summary>

#### Latency testing

Create a detailed timing format file for `curl` to help measure various latency metrics including DNS lookup, connection time, and total response time for precise performance analysis.

</details>

```python
# Create curl-format.txt with the following content:
curl_format_content = """     time_namelookup:  %{time_namelookup}
        time_connect:  %{time_connect}
     time_appconnect:  %{time_appconnect}
    time_pretransfer:  %{time_pretransfer}
       time_redirect:  %{time_redirect}
  time_starttransfer:  %{time_starttransfer}
                     ----------
          time_total:  %{time_total}
          http_code:  %{http_code}
       size_download:  %{size_download}
      speed_download:  %{speed_download}"""

with open('curl-format.txt', 'w') as f:
    f.write(curl_format_content)

print("curl-format.txt has been created successfully!")
```

<!-- cell:53 -->

现在，使用 `curl` 和详细的计时格式来测量单个请求的精确响应时间。这提供了对请求处理管道的每个步骤的精细洞察。

<details markdown="1">
<summary>查看英文原文 · 单元 53</summary>

Now use `curl` with your detailed timing format to measure the precise response times for a single request. This provides granular insights into each step of the request processing pipeline.

</details>

```python
# Measure response times
!curl -w "@curl-format.txt" -X POST http://localhost:8890/v1/chatqna \
  -H "Content-Type: application/json" \
  -d '{"messages": "What is machine learning?"}'
```

<!-- cell:55 -->

<a id="step-4-evaluation-of-results"></a>

### 步骤 4：结果评估

评估结果包括以下内容：

- **响应时间**：平均值、中位数和第 95 个百分位
- **吞吐量**：每秒请求数
- **准确性**：答案质量指标
- **资源使用**：CPU、GPU 和内存消耗

<details markdown="1">
<summary>查看英文原文 · 单元 55</summary>

### Step 4: Evaluation of results

The evaluation results include the following:

- **Response time**: Average, median, and 95th percentile
- **Throughput**: Requests per second
- **Accuracy**: Answer quality metrics
- **Resource usage**: CPU, GPU, and memory consumption

</details>

<!-- cell:56 -->

<a id="common-issues-and-solutions"></a>

## 常见问题及解决方案

您收集的性能结果可能表明 ChatQnA 系统存在某些性能问题。

<a id="gpu-memory-errors"></a>

### GPU 内存错误

**症状**：内存不足或类似错误。

**解决方案**：减少批大小。

<details markdown="1">
<summary>查看英文原文 · 单元 56</summary>

## Common issues and solutions

The performance results you collected could potentially indicate certain performance issues with the ChatQnA system.

### GPU memory errors

**Symptoms**: Out-of-memory or similar errors.

**Solution**: Reduce the batch size.

</details>

```python
# Reduce batch size in vLLM configuration
# Edit compose_vllm.yaml, modify vLLM service command:
--max-model-len 2048 --tensor-parallel-size 1
```

<!-- cell:58 -->

<a id="service-startup-failures"></a>

### 服务启动失败

**现象**：服务无法启动或停留在`starting`状态。

**解决办法**：检查日志。

<details markdown="1">
<summary>查看英文原文 · 单元 58</summary>

### Service startup failures

**Symptoms**: Services fail to start or remain in the `starting` state.

**Solution**: Check the logs.

</details>

```python
# Check logs for specific errors
!docker compose -f compose_vllm.yaml logs
```

<!-- cell:60 -->

通过将 `restart-vllm` 传递给 `run_chatqna.sh` 脚本来重新启动非活动服务。

<details markdown="1">
<summary>查看英文原文 · 单元 60</summary>

Restart inactive services by passing `restart-vllm` to the `run_chatqna.sh` script.

</details>

```python
# Restart services
!./run_chatqna.sh restart-vllm
```

<!-- cell:62 -->

<a id="redis-index-issues"></a>

### Redis索引问题

**现象**：检索服务找不到文档。

**解决方案**：修复Redis索引。

<details markdown="1">
<summary>查看英文原文 · 单元 62</summary>

### Redis index issues

**Symptoms**: The retrieval service fails to find documents.

**Solution**: Fix the Redis index.

</details>

```python
!./fix_redis_index.sh
```

<!-- cell:64 -->

然后手动重新创建Redis索引。

<details markdown="1">
<summary>查看英文原文 · 单元 64</summary>

Then recreate the Redis index manually.

</details>

```python

!docker exec chatqna-redis-vector-db redis-cli FT.CREATE rag-redis ON HASH PREFIX 1 doc: SCHEMA content TEXT WEIGHT 1.0 distance NUMERIC
```

<!-- cell:66 -->

<a id="model-download-failures"></a>

### 模型下载失败

**症状**：服务无法下载模型。

**解决方案**：验证 Hugging Face 令牌。

<details markdown="1">
<summary>查看英文原文 · 单元 66</summary>

### Model download failures

**Symptoms**: Services fail to download models.

**Solution**: Verify the Hugging Face token.

</details>

```python
# Check HF token
!echo $CHATQNA_HUGGINGFACEHUB_API_TOKEN
```

<!-- cell:68 -->

手动设置您的Hugging Face标记。

<details markdown="1">
<summary>查看英文原文 · 单元 68</summary>

Set your Hugging Face token manually.

</details>

```python
# Set token manually
!export CHATQNA_HUGGINGFACEHUB_API_TOKEN="your_token_here"
```

<!-- cell:70 -->

<a id="advanced-configuration"></a>

## 高级配置

本节介绍高级场景，例如如何使用不同的模型。

<a id="custom-model-configuration"></a>

### 自定义模型配置

编辑 `set_env_vllm.sh` 文件以使用不同的模型。

运行以下命令更改 LLM 模型：

<details markdown="1">
<summary>查看英文原文 · 单元 70</summary>

## Advanced configuration

This section covers advanced scenarios, such as how to use different models.

### Custom model configuration

Edit the `set_env_vllm.sh` file to use different models.

Run this command to change the LLM model:

</details>

```python
!export CHATQNA_LLM_MODEL_ID="Qwen/Qwen2.5-14B-Instruct"
```

<!-- cell:72 -->

运行以下命令来更改嵌入模型：

<details markdown="1">
<summary>查看英文原文 · 单元 72</summary>

Run this command to change the embedding model:

</details>

```python
!export CHATQNA_EMBEDDING_MODEL_ID="BAAI/bge-large-en-v1.5"
```

<!-- cell:74 -->

运行此命令以更改重新排名模型：

<details markdown="1">
<summary>查看英文原文 · 单元 74</summary>

Run this command to change the reranking model:

</details>

```python
!export CHATQNA_RERANK_MODEL_ID="BAAI/bge-reranker-large"
```

<!-- cell:76 -->

<a id="troubleshooting"></a>

## 故障排除

如果系统未按预期工作，请使用本节作为参考。

<a id="diagnostic-commands"></a>

### 诊断命令

运行以下命令检查系统资源：

<details markdown="1">
<summary>查看英文原文 · 单元 76</summary>

## Troubleshooting

Use this section as a reference if the system isn't working as expected.

### Diagnostic commands

Run this command to check the system resources:

</details>

```python
!./detect_issues.sh
```

<!-- cell:78 -->

使用以下命令对整个系统运行快速测试：

<details markdown="1">
<summary>查看英文原文 · 单元 78</summary>

Run a quick test for the whole system using this command:

</details>

```python
!./quick_test_chatqna.sh eval-only
```

<!-- cell:80 -->

运行以下命令来检查服务运行状况：

<details markdown="1">
<summary>查看英文原文 · 单元 80</summary>

Run this command to check the service health:

</details>

```python
!docker compose -f compose_vllm.yaml ps
```

<!-- cell:82 -->

<a id="log-analysis"></a>

### 日志分析

<details markdown="1">
<summary>查看英文原文 · 单元 82</summary>

### Log analysis

</details>

<!-- cell:83 -->

查看所有日志：

<details markdown="1">
<summary>查看英文原文 · 单元 83</summary>

View all logs:

</details>

```python
!docker compose -f compose_vllm.yaml logs
```

<!-- cell:85 -->

查看具体服务日志：

<details markdown="1">
<summary>查看英文原文 · 单元 85</summary>

Review specific service logs:

</details>

```python
!docker compose -f compose_vllm.yaml logs -f chatqna-vllm-service
```

<!-- cell:87 -->

检查是否有错误：

<details markdown="1">
<summary>查看英文原文 · 单元 87</summary>

Check for errors:

</details>

```python
!docker compose -f compose_vllm.yaml logs | grep -i error
```

<!-- cell:89 -->

<a id="gpu-memory-management"></a>

### GPU内存管理

确定 GPU 内存使用情况并根据需要清除内存。

####检查GPU内存状态

<details markdown="1">
<summary>查看英文原文 · 单元 89</summary>

### GPU memory management

Determine GPU memory usage and clear memory if required.

#### Check GPU memory status

</details>

<!-- cell:90 -->

检查当前 GPU 内存使用情况。

**注意**：对于 ROCm 6.4 及更早版本，请改用 `rocm-smi` 命令。

<details markdown="1">
<summary>查看英文原文 · 单元 90</summary>

Check the current GPU memory usage.

**Note**: For ROCm 6.4 and earlier, use the `rocm-smi` command instead.

</details>

```python
# Check current GPU memory usage
# Expected output shows VRAM% and GPU% usage
# If VRAM% is high (>80%) but GPU% is low, memory may be fragmented
!amd-smi
```

<!-- cell:92 -->

<a id="clear-gpu-memory-if-necessary"></a>

#### 清除 GPU 内存（如有必要）

如果您遇到 GPU 内存问题或 VRAM 使用率高而 GPU 利用率低，请尝试以下部分中的命令：

**选项 1：终止 GPU 进程**

<details markdown="1">
<summary>查看英文原文 · 单元 92</summary>

#### Clear GPU memory (if necessary)

If you encounter GPU memory issues or high VRAM usage with low GPU utilization, try the commands in the following sections:

**Option 1: Kill GPU processes**

</details>

<!-- cell:93 -->

查找正在使用 GPU 的所有进程：

<details markdown="1">
<summary>查看英文原文 · 单元 93</summary>

Find any processes that are using the GPU:

</details>

```python
!sudo fuser -v /dev/kfd
```

<!-- cell:95 -->

杀死GPU相关进程：

<details markdown="1">
<summary>查看英文原文 · 单元 95</summary>

Kill the GPU-related processes:

</details>

```python

!sudo pkill -f "python|vllm|docker"
```

<!-- cell:97 -->

**选项 2：重新启动 GPU 服务**

<details markdown="1">
<summary>查看英文原文 · 单元 97</summary>

**Option 2: Restart GPU services**

</details>

<!-- cell:98 -->

重启`amdgpu`及相关服务：

<details markdown="1">
<summary>查看英文原文 · 单元 98</summary>

Restart the `amdgpu` and related services:

</details>

```python
!sudo systemctl restart amdgpu
!sudo systemctl restart kfd
```

<!-- cell:100 -->

**选项 3：系统重启**

<details markdown="1">
<summary>查看英文原文 · 单元 100</summary>

**Option 3: System reboot**

</details>

<!-- cell:101 -->

如果其他方法不起作用，请重新启动系统。这是处理 GPU 内存问题最可靠的方法。

<details markdown="1">
<summary>查看英文原文 · 单元 101</summary>

If the other methods don't work, reboot the system. This is the most reliable way of dealing with GPU memory issues.

</details>

```python
# Note: If you're on a remote server, wait approximately 30 seconds to 1 minute
# before attempting to SSH back into the server
!sudo reboot
```

<!-- cell:103 -->

清除 GPU 内存后，验证内存是否再次可用。

**注意**：对于 ROCm 6.4 及更早版本，请改用 `rocm-smi` 命令。

<details markdown="1">
<summary>查看英文原文 · 单元 103</summary>

After clearing GPU memory, verify memory is available again.

**Note**: For ROCm 6.4 and earlier, use the `rocm-smi` command instead.

</details>

```python

# Check GPU memory is now available
# Expected: VRAM% should be low (<20%) and GPU% should be 0%
!amd-smi
```

<!-- cell:105 -->

<a id="conclusion"></a>

## 结论

本教程提供了在 AMD GPU 上使用 vLLM 部署 ChatQnA 并执行详细性能评估的综合指南。 ChatQnA 系统提供：

- **高性能**：vLLM 优化的 推理
- **可扩展性**：基于 Docker 的微服务架构
- **监控**：内置性能指标
- **灵活性**：可配置模型和参数

如需其他支持或高级配置，请参阅[项目文档](https://github.com/opea-project/GenAIExamples/tree/main/ChatQnA) 或在存储库中创建问题。

<a id="next-steps"></a>

### 后续步骤

1. **自定义模型**：尝试不同的 LLM 和嵌入模型。
2. **规模部署**：为更高的吞吐量添加多个GPU节点。
3. **优化性能**：针对您的特定用例微调 vLLM 参数。
4. **监控生产**：为生产部署设置全面监控。

**注意**：本教程假设您拥有必要的权限并且已安装所有必需的软件。对于生产部署，请考虑额外的安全措施和监控解决方案。

<details markdown="1">
<summary>查看英文原文 · 单元 105</summary>

## Conclusion

This tutorial provides a comprehensive guide for deploying ChatQnA with vLLM on AMD GPUs and performing detailed performance evaluation. The ChatQnA system offers:

- **High performance**: vLLM-optimized inference
- **Scalability**: Docker-based microservices architecture
- **Monitoring**: Built-in performance metrics
- **Flexibility**: Configurable models and parameters

For additional support or advanced configurations, see the [project documentation](https://github.com/opea-project/GenAIExamples/tree/main/ChatQnA) or create an issue in the repository.

### Next steps

1. **Customize models**: Experiment with different LLM and embedding models.
2. **Scale deployment**: Add multiple GPU nodes for higher throughput.
3. **Optimize performance**: Fine-tune vLLM parameters for your specific use case.
4. **Monitor production**: Set up comprehensive monitoring for production deployments.

**Note**: This tutorial assumes you have the necessary permissions and that all required software is installed. For production deployments, consider additional security measures and monitoring solutions.

</details>
