> 原文作者：Yu Wang  
> [AMD 官方教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/opea_deployment_and_evaluation.html) · [固定版本源码](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/inference/opea_deployment_and_evaluation.ipynb)  
> 上游版本：`0071bfd4bd3bd9525405c09a6ae69c983b09312a` · 同步日期：2026-10-12  
> 翻译状态：机器初译，待逐篇人工校对；代码与命令保持原文，未在 AMD GPU 上运行验证。

[中文 Notebook](../../notebooks/inference/opea_deployment_and_evaluation.ipynb) · [英文原文](../../original/inference/opea_deployment_and_evaluation.md) · [中英对照](../../bilingual/inference/opea_deployment_and_evaluation.md)

---

<a id="chatqna-vllm-deployment-and-performance-evaluation"></a>

# ChatQnA：vLLM 部署与性能评估
**作者**：Yu Wang
**知识水平**：初学者

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

<a id="deployment-guide"></a>

## 部署指南

要部署 ChatQnA，请按照下列步骤操作：

<a id="step-1-pull-the-source-code-from-github"></a>

### 第 1 步：从 GitHub 获取源代码

首先，克隆企业 AI 开放平台 (OPEA) GenAIExamples 存储库，其中包含 ChatQnA 实现和部署所需的其他 AI 示例。

```python
# Home directory
import os
HOME_DIR = os.getcwd()
# Open Platform for Enterprise AI (OPEA)
!git clone https://github.com/opea-project/GenAIExamples.git
```

接下来，克隆 LaunchPad 存储库，该存储库提供专为 AMD GPU 环境上的 ChatQnA 使用案例设计的一键部署脚本和配置文件。

```python
# One click deployment scripts for the use case
!git clone https://github.com/Yu-amd/LaunchPad.git
```

最后，克隆 GenAIEval 评估存储库，其中包含您将用于评估 ChatQnA 系统性能的基准测试工具。

```python
# Pipeline performance evaluation harness
!git clone https://github.com/opea-project/GenAIEval.git
```

LaunchPad 项目使用与 OPEA 项目相同的层次结构。将每个目录中的 LaunchPad 脚本和 YAML 文件复制到 OPEA 文件夹中的相应目录。

```python
# Copy necessary scripts and configuration files to the OPEA directory
!cp {HOME_DIR}/LaunchPad/GenAIExamples/ChatQnA/docker_compose/amd/gpu/rocm/*.sh {HOME_DIR}/GenAIExamples/ChatQnA/docker_compose/amd/gpu/rocm/
!cp {HOME_DIR}/LaunchPad/GenAIExamples/ChatQnA/docker_compose/amd/gpu/rocm/*.yaml {HOME_DIR}/GenAIExamples/ChatQnA/docker_compose/amd/gpu/rocm/
!cp {HOME_DIR}/LaunchPad/GenAIExamples/ChatQnA/docker_compose/amd/gpu/rocm/.env {HOME_DIR}/GenAIExamples/ChatQnA/docker_compose/amd/gpu/rocm/
!cp -r {HOME_DIR}/LaunchPad/GenAIEval/evals/benchmarks/* {HOME_DIR}/GenAIEval/evals/benchmark/
```

<a id="step-2-environment-setup"></a>

### 第 2 步：环境设置

现在导航到所有配置文件和脚本所在的 OPEA 部署目录。

```python
# Navigate to the OPEA deployment directory
%cd {HOME_DIR}/GenAIExamples/ChatQnA/docker_compose/amd/gpu/rocm
```

您需要配置环境变量管理。首先，安装 `python-dotenv` 软件包，它允许您从 `.env` 文件加载环境变量。然后导入必要的模块并从包含配置设置的文件中加载环境变量。

您还需要配置 Hugging Face API 令牌，这是下载 ChatQnA 系统使用的 AI 模型所必需的。将 `your_token_here` 替换为实际的 Hugging Face 令牌。

```python
# Install and load .env
!pip install python-dotenv
# Configure Hugging Face token
!sed -i 's/CHATQNA_HUGGINGFACEHUB_API_TOKEN=""/CHATQNA_HUGGINGFACEHUB_API_TOKEN="YOUR_ACTUAL_TOKEN_HERE"/' .env
```

设置所有环境变量。

```python
# Load environment variables
from dotenv import load_dotenv
import os
load_dotenv()  # Loads variables from .env file
```

现在使用提供的脚本设置 vLLM 环境。这会在 AMD GPU 上配置高性能 LLM 推理 所需的所有组件。

```python
# Setup vLLM environment
!./run_chatqna.sh setup-vllm
```

<a id="step-3-deploy-the-workload"></a>

### 步骤 3：部署工作负载

配置好环境后，您现在可以启动 vLLM 服务。这将启动 ChatQnA 系统所需的所有容器和服务。

```python
# Start vLLM services
import subprocess
subprocess.run(["./run_chatqna.sh", "start-vllm"], check=True)
```

检查所有正在运行的服务的状态，以确保它们正确启动并正常运行。

```python
# Check service status
!./run_chatqna.sh status
```

监视 vLLM 服务日志 60 秒以验证服务是否正确启动。查看日志中是否有任何初始化问题。

```python
# Check chatqna-vllm-service status
!timeout 200 docker logs -f chatqna-vllm-service
```

<a id="step-4-verify-the-deployment"></a>

### 步骤 4：验证部署

通过检查其状态和端口映射来验证所有 Docker 容器是否正常运行。

```python
# Check running containers
!docker ps
```

下一个命令向后端 API 发送一条简单的测试消息，以验证 ChatQnA 服务是否正常工作。

```python
# Test backend API
!curl -X POST http://localhost:8890/v1/chatqna \
  -H "Content-Type: application/json" \
  -d '{"messages": "Hello, how are you?"}'
```

<a id="step-5-upload-documents"></a>

### 第5步：上传文件

创建示例文档并将其上传到系统。这演示了如何将文档输入 ChatQnA 系统以进行检索和问答。

```python
# Create a text file
!echo "Your document content here" > document.txt

# Upload the file
!curl -X POST http://localhost:18104/v1/dataprep/ingest \
  -H "Content-Type: multipart/form-data" \
  -F "files=@document.txt"
```

现在通过检查 Redis 矢量数据库的内容来验证文档是否已成功上传并建立索引。

```python
# Verify the upload worked
# Check if the document was indexed
!curl -X POST http://localhost:18104/v1/dataprep/get \
  -H "Content-Type: application/json" \
  -d '{"index_name": "rag-redis"}'
```

您还可以一次上传多个文档。以下介绍了如何同时创建和上传多个文档以建立您的知识库。

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

```python
# Navigate to evaluation directory
%cd {HOME_DIR}/GenAIEval
```

安装评估工具所需的依赖项并在开发模式下设置 GenAIEval 包。

```python
# Install evaluation dependencies
!pip install -r requirements.txt
!pip install -e .
```

<a id="step-2-run-the-basic-evaluation"></a>

### 第 2 步：运行基本评估

现在导航回 ChatQnA 部署目录并在已部署的系统上运行性能评估测试。

```python
# Navigate back to GenAIExamples/ChatQnA/docker_compose/amd/gpu/rocm/
%cd {HOME_DIR}/GenAIExamples/ChatQnA/docker_compose/amd/gpu/rocm/
```

现在运行 vLLM 评估脚本，该脚本测试 ChatQnA 系统的性能，测量 吞吐量、延迟 等指标和响应质量。

```python
# Run vLLM evaluation
!./run_chatqna.sh vllm-eval
```

<a id="step-3-performance-metrics"></a>

### 步骤 3：绩效指标

本部分执行额外的 吞吐量 和 延迟 测试。

<a id="throughput-testing"></a>

#### 吞吐量 测试

安装 Apache Bench (ab)，它执行负载测试并测量各种条件下 ChatQnA API 的 吞吐量。

```python
# Install dependency
!apt install -y apache2-utils
```

创建包含复杂问题的测试文件，以评估系统处理详细的多部分查询并生成全面响应的能力。

```python
# Create a complex test file
!echo '{"messages": "Can you provide a detailed explanation of how neural networks work, including the concepts of forward propagation, backpropagation, and gradient descent? Also explain how these concepts relate to deep learning and why they are important for modern AI systems."}' > test_data.json
```

现在使用 Apache Bench 运行负载测试，模拟 100 个并发请求和 10 个并发连接。这可以衡量系统的 吞吐量 和压力下的性能。

```python
# Test concurrent requests
!ab -n 100 -c 10 -p test_data.json -T application/json \
  http://localhost:8890/v1/chatqna
```

<a id="latency-testing"></a>

#### 延迟 测试

为 `curl` 创建详细的计时格式文件，以帮助测量各种 延迟 指标，包括 DNS 查找、连接时间和总响应时间，以进行精确的性能分析。

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

现在，使用 `curl` 和详细的计时格式来测量单个请求的精确响应时间。这提供了对请求处理管道的每个步骤的精细洞察。

```python
# Measure response times
!curl -w "@curl-format.txt" -X POST http://localhost:8890/v1/chatqna \
  -H "Content-Type: application/json" \
  -d '{"messages": "What is machine learning?"}'
```

<a id="step-4-evaluation-of-results"></a>

### 步骤 4：结果评估

评估结果包括以下内容：

- **响应时间**：平均值、中位数和第 95 个百分位
- **吞吐量**：每秒请求数
- **准确性**：答案质量指标
- **资源使用**：CPU、GPU 和内存消耗

<a id="common-issues-and-solutions"></a>

## 常见问题及解决方案

您收集的性能结果可能表明 ChatQnA 系统存在某些性能问题。

<a id="gpu-memory-errors"></a>

### GPU 内存错误

**症状**：内存不足或类似错误。

**解决方案**：减少批大小。

```python
# Reduce batch size in vLLM configuration
# Edit compose_vllm.yaml, modify vLLM service command:
--max-model-len 2048 --tensor-parallel-size 1
```

<a id="service-startup-failures"></a>

### 服务启动失败

**现象**：服务无法启动或停留在`starting`状态。

**解决办法**：检查日志。

```python
# Check logs for specific errors
!docker compose -f compose_vllm.yaml logs
```

通过将 `restart-vllm` 传递给 `run_chatqna.sh` 脚本来重新启动非活动服务。

```python
# Restart services
!./run_chatqna.sh restart-vllm
```

<a id="redis-index-issues"></a>

### Redis索引问题

**现象**：检索服务找不到文档。

**解决方案**：修复Redis索引。

```python
!./fix_redis_index.sh
```

然后手动重新创建Redis索引。

```python

!docker exec chatqna-redis-vector-db redis-cli FT.CREATE rag-redis ON HASH PREFIX 1 doc: SCHEMA content TEXT WEIGHT 1.0 distance NUMERIC
```

<a id="model-download-failures"></a>

### 模型下载失败

**症状**：服务无法下载模型。

**解决方案**：验证 Hugging Face 令牌。

```python
# Check HF token
!echo $CHATQNA_HUGGINGFACEHUB_API_TOKEN
```

手动设置您的Hugging Face标记。

```python
# Set token manually
!export CHATQNA_HUGGINGFACEHUB_API_TOKEN="your_token_here"
```

<a id="advanced-configuration"></a>

## 高级配置

本节介绍高级场景，例如如何使用不同的模型。

<a id="custom-model-configuration"></a>

### 自定义模型配置

编辑 `set_env_vllm.sh` 文件以使用不同的模型。

运行以下命令更改 LLM 模型：

```python
!export CHATQNA_LLM_MODEL_ID="Qwen/Qwen2.5-14B-Instruct"
```

运行以下命令来更改嵌入模型：

```python
!export CHATQNA_EMBEDDING_MODEL_ID="BAAI/bge-large-en-v1.5"
```

运行此命令以更改重新排名模型：

```python
!export CHATQNA_RERANK_MODEL_ID="BAAI/bge-reranker-large"
```

<a id="troubleshooting"></a>

## 故障排除

如果系统未按预期工作，请使用本节作为参考。

<a id="diagnostic-commands"></a>

### 诊断命令

运行以下命令检查系统资源：

```python
!./detect_issues.sh
```

使用以下命令对整个系统运行快速测试：

```python
!./quick_test_chatqna.sh eval-only
```

运行以下命令来检查服务运行状况：

```python
!docker compose -f compose_vllm.yaml ps
```

<a id="log-analysis"></a>

### 日志分析

查看所有日志：

```python
!docker compose -f compose_vllm.yaml logs
```

查看具体服务日志：

```python
!docker compose -f compose_vllm.yaml logs -f chatqna-vllm-service
```

检查是否有错误：

```python
!docker compose -f compose_vllm.yaml logs | grep -i error
```

<a id="gpu-memory-management"></a>

### GPU内存管理

确定 GPU 内存使用情况并根据需要清除内存。

####检查GPU内存状态

检查当前 GPU 内存使用情况。

**注意**：对于 ROCm 6.4 及更早版本，请改用 `rocm-smi` 命令。

```python
# Check current GPU memory usage
# Expected output shows VRAM% and GPU% usage
# If VRAM% is high (>80%) but GPU% is low, memory may be fragmented
!amd-smi
```

<a id="clear-gpu-memory-if-necessary"></a>

#### 清除 GPU 内存（如有必要）

如果您遇到 GPU 内存问题或 VRAM 使用率高而 GPU 利用率低，请尝试以下部分中的命令：

**选项 1：终止 GPU 进程**

查找正在使用 GPU 的所有进程：

```python
!sudo fuser -v /dev/kfd
```

杀死GPU相关进程：

```python

!sudo pkill -f "python|vllm|docker"
```

**选项 2：重新启动 GPU 服务**

重启`amdgpu`及相关服务：

```python
!sudo systemctl restart amdgpu
!sudo systemctl restart kfd
```

**选项 3：系统重启**

如果其他方法不起作用，请重新启动系统。这是处理 GPU 内存问题最可靠的方法。

```python
# Note: If you're on a remote server, wait approximately 30 seconds to 1 minute
# before attempting to SSH back into the server
!sudo reboot
```

清除 GPU 内存后，验证内存是否再次可用。

**注意**：对于 ROCm 6.4 及更早版本，请改用 `rocm-smi` 命令。

```python

# Check GPU memory is now available
# Expected: VRAM% should be low (<20%) and GPU% should be 0%
!amd-smi
```

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
