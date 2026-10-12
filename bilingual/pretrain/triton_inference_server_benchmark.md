> 原文作者：Lin Sun, Dipto Deb  
> [AMD 官方教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/triton_inference_server_benchmark.html) · [固定版本源码](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/pretrain/triton_inference_server_benchmark.ipynb)  
> 上游版本：`0071bfd4bd3bd9525405c09a6ae69c983b09312a` · 同步日期：2026-10-12  
> 翻译状态：机器初译，待逐篇人工校对；代码与命令保持原文，未在 AMD GPU 上运行验证。

<!-- cell:1 -->

<!--
  SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
  SPDX-License-Identifier: Apache-2.0
-->

<a id="training-and-serving-a-finalnet-ctr-model-with-triton-inference-server"></a>

# 使用 Triton Inference Server 训练与部署 FinalNet CTR 模型
**作者**：Lin Sun, Dipto Deb
**知识水平**：初学者  

本教程演示如何使用生产级模型服务器 [Triton 推理 服务器](https://github.com/ROCm/triton-inference-server-server/tree/rocm7.2_r25.12) 的 ROCm 端口在 AMD Instinct™ GPU 上为高性能 推理 提供经过训练的模型。它展示了如何使用 AMD 的 [ROCm™](https://rocm.docs.amd.com/) 软件堆栈和 MIGraphX 执行提供程序部署模型，然后测量模型的运行速度。

为了使示例完整且可重现，本教程使用名为 **FinalNet** 的真实**点击率 (CTR) 预测模型**，并引导您完成端到端的过程：首先对模型进行训练，然后进行部署和基准测试。模型训练不是这里的重点，只是为了使教程独立。如果您已经有经过训练的模型，则可以直接跳至部署；请参阅[已经有训练好的模型？](#already-have-a-trained-model)。

本教程假设您之前没有使用 Triton 推理 服务器、ONNX 或推荐模型的经验。如果本教程中的某些术语对您来说很陌生，请不要担心 - [关键概念](#key-concepts) 部分将以通俗易懂的语言解释这些术语。

> **配套博客文章：** 本教程附带 AMD ROCm 博客 [*使用 ONNX 运行时后端通过 Triton 推理 服务器提供 CTR 推荐模型*](https://rocm.blogs.amd.com/artificial-intelligence/triton-inference-server/README.html)，其中介绍了 Triton 推理 服务器的 ROCm 版本中的 ONNX 运行时和 Python 后端支持。请参阅它以了解更多背景信息和更广泛的受支持工作负载。

<a id="what-you-will-learn"></a>

### 你将学到什么

**核心技能**：在 AMD GPU 上部署 推理 模型并对其进行基准测试。

- 如何获取启用 ROCm 的 Triton 推理 服务器容器映像。
- 如何使用由 **MIGraphX 执行提供程序**加速的 **ONNX 运行时后端** 在 Triton 推理 服务器上部署 ONNX 模型。
- 如何使用 `perf_analyzer` 在您自己的硬件上测量服务器的 吞吐量 和 延迟，以及如何解释结果。

**可选技能**：生成和部署模型。

- 如何在 PyTorch ROCm 容器内的公共 Criteo\_x4 数据集上训练 FinalNet CTR 模型。
- 如何将经过训练的 PyTorch 模型导出为便携式 **ONNX** 格式。

<a id="workflow-at-a-glance"></a>

### 工作流程概览

- **[步骤1](#step-1-get-the-triton-inference-server-docker-image)：获取服务器镜像**。从 Docker Hub 中拉取 Triton 推理 服务器镜像的预构建 ROCm 分支。
- **[步骤 2](#step-2-train-finalnet-on-amd-gpus)：训练** *（可选）*。在 Criteo\_x4 数据集上训练 FinalNet。
- **[步骤 3](#step-3-export-the-trained-model-to-onnx)：导出** *（可选）*。将训练好的 检查点 转换为 ONNX 格式。
- **[步骤 4](#step-4-deploy-finalnet-with-the-migraphx-execution-provider)：部署**。使用 MIGraphX 执行提供程序在 Triton 推理 服务器上提供 ONNX 模型。
- **[步骤 5](#step-5-measure-serving-performance-with-perf_analyzer) 和 [步骤 6](#step-6-plot-your-own-results)：测量**。针对正在运行的服务器运行客户端基准测试。

<a id="already-have-a-trained-model"></a>

### 已经有经过训练的模型？

这是一个端到端演示，部署和基准测试步骤是核心，培训和导出步骤是可选的。 **[步骤 1](#step-1-get-the-triton-inference-server-docker-image)** 之后，如果您已经有：
- **经过训练的 FinalNet 检查点** → 跳过 **步骤 2** 并从 **[步骤 3](#step-3-export-the-trained-model-to-onnx)** 开始将其导出到 ONNX。
- **ONNX 模型**（FinalNet 或您自己的）→ 完全跳过**步骤 2 和步骤 3**。运行 **[环境设置](#environment-setup)** 中的单元来设置 `WORKSPACE` 并创建目录布局，将模型复制到 `${WORKSPACE}/model_repository/FinalNet_onnx/1/model.onnx`，然后直接跳到 **[步骤 4](#step-4-deploy-finalnet-with-the-migraphx-execution-provider)** 进行部署（如果您的模型*不是* FinalNet，还可以编辑输入/输出名称和形状`config.pbtxt` 以匹配您的型号）。

所有命令都设计为在 **主机** 计算机上运行。需要训练环境的步骤由 `docker exec` 路由到容器中。

<details markdown="1">
<summary>查看英文原文 · 单元 1</summary>

<!--
  SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
  SPDX-License-Identifier: Apache-2.0
-->

# Training and serving a FinalNet CTR model with Triton Inference Server

**Authors**: Lin Sun, Dipto Deb  
**Knowledge level**: Beginner  

This tutorial demonstrates how to serve a trained model for high-performance inference on AMD Instinct™ GPUs using the ROCm port of [Triton Inference Server](https://github.com/ROCm/triton-inference-server-server/tree/rocm7.2_r25.12), a production-grade model server. It shows how to deploy a model with AMD's [ROCm™](https://rocm.docs.amd.com/) software stack and the MIGraphX execution provider and then measure how fast the model runs.

To make the example complete and reproducible, this tutorial uses a real **click-through rate (CTR) prediction model** called **FinalNet** and walks you through the process end-to-end: the model is trained first, then deployed and benchmarked. Model training is not the focus here, but is included only to make the tutorial self-contained. If you already have a trained model, you will be able to skip straight to deployment; see [Already have a trained model?](#already-have-a-trained-model).

This tutorial assumes no prior experience with Triton Inference Server, ONNX, or the recommendation models. If some of the terms in this tutorial are new to you, don't worry—the [Key concepts](#key-concepts) section will explain the terms in plain language.

> **Companion blog post:** This tutorial accompanies the AMD ROCm blog [*Serving CTR Recommendation Models with Triton Inference Server using the ONNX Runtime Backend*](https://rocm.blogs.amd.com/artificial-intelligence/triton-inference-server/README.html), which introduces ONNX Runtime and Python backend support in the ROCm build of Triton Inference Server. Refer to it for additional background and a broader set of supported workloads.

### What you will learn

**Core skills**: Deploying and benchmarking a model for inference on AMD GPUs.

- How to obtain a ROCm-enabled Triton Inference Server container image.
- How to deploy an ONNX model on Triton Inference Server using the **ONNX Runtime backend** accelerated by the **MIGraphX execution provider**.
- How to measure the server's throughput and latency on your own hardware with `perf_analyzer`, and how to interpret the results.

**Optional skills**: Produce and deploy a model.

- How to train the FinalNet CTR model on the public Criteo\_x4 dataset inside a PyTorch ROCm container.
- How to export a trained PyTorch model to the portable **ONNX** format.

### Workflow at a glance

- **[Step 1](#step-1-get-the-triton-inference-server-docker-image): Get the server image**. Pull the prebuilt ROCm fork of Triton Inference Server image from Docker Hub.
- **[Step 2](#step-2-train-finalnet-on-amd-gpus): Train** *(optional)*. Train FinalNet on the Criteo\_x4 dataset.
- **[Step 3](#step-3-export-the-trained-model-to-onnx): Export** *(optional)*. Convert the trained checkpoint to the ONNX format.
- **[Step 4](#step-4-deploy-finalnet-with-the-migraphx-execution-provider): Deploy**. Serve the ONNX model on Triton Inference Server with the MIGraphX execution provider.
- **[Step 5](#step-5-measure-serving-performance-with-perf_analyzer) and [Step 6](#step-6-plot-your-own-results): Measure**. Run a client-side benchmark against the running server.

### Already have a trained model?

This is an end-to-end demo, with the deployment and benchmark steps being core and the training and export steps being optional. After **[Step 1](#step-1-get-the-triton-inference-server-docker-image)**, if you already have:

- **A trained FinalNet checkpoint** → skip **Step 2** and start at **[Step 3](#step-3-export-the-trained-model-to-onnx)** to export it to ONNX.
- **An ONNX model** (FinalNet or your own) → skip **Step 2 and Step 3** entirely. Run the cells in **[Environment setup](#environment-setup)** to set `WORKSPACE` and create the directory layout, copy your model to `${WORKSPACE}/model_repository/FinalNet_onnx/1/model.onnx`, then jump straight to **[Step 4](#step-4-deploy-finalnet-with-the-migraphx-execution-provider)** to deploy (if your model is *not* FinalNet, also edit the input/output names and shapes in `config.pbtxt` to match your model).

All commands are designed to be run on the **host** machine. Steps that need the training environment are routed into a container by `docker exec`.

</details>

<!-- cell:2 -->

<a id="key-concepts"></a>

## 关键概念

本节介绍本教程涉及的主要技术概念。

<a id="click-through-rate-ctr-prediction"></a>

### 点击率 (CTR) 预测

**点击率预测**估计用户点击特定项目（广告、产品或推荐）的概率。它是经济上最重要的机器学习问题之一，为推荐系统和在线广告提供动力。输入通常是**数字特征**（例如计数）和**分类特征**（例如设备类型或用户 ID）的混合，输出是 0 到 1 之间的单个概率。

<a id="finalnet"></a>

### 最终网

**[FinalNet](https://arxiv.org/abs/2301.01624)** 是 SIGIR 2023 上推出的用于 CTR 预测的神经网络架构。它目前位于 [BARS Criteo\_x4 排行榜](https://openbenchmark.github.io/BARS/CTR/leaderboard/criteo_x4.html) 的顶部，这是一个在标准化数据集上对 CTR 模型进行排名的社区基准。它在这里用作现实的、与生产相关的架构来训练和服务 CTR 模型。

<a id="the-criteo_x4-dataset"></a>

### Criteo\_x4 数据集

**[Criteo\_x4]（https://huggingface.co/datasets/reczoo/Criteo_x4)** 是一个大型、公开的广告数据集，广泛用于基准 CTR 模型。每一行描述一次广告印象，包含 13 个数字和 26 个分类特征，以及一个指示广告是否被点击的标签。它是公开可用的，不需要下载访问令牌。

<a id="triton-inference-server"></a>

### Triton 推理 服务器

**[Triton 推理 服务器]（https://github.com/ROCm/triton-inference-server-server/tree/rocm7.2_r25.12)** 是一个开源模型服务系统。您向它提供一个经过训练的模型，它会公开客户端调用以运行 推理 的 HTTP/gRPC 端点，同时为您处理批处理、并发和多种模型格式。本教程使用 Triton 推理 服务器的 **ROCm 分支**，这增加了对 AMD GPU 的支持。

<a id="onnx-and-the-onnx-runtime-backend"></a>

### ONNX 和 ONNX 运行时后端

**[ONNX（开放神经网络交换）](https://onnx.ai/)** 是一种可移植、独立于框架的文件格式，用于训练模型。将 PyTorch 模型导出到 ONNX 允许 Triton 推理 服务器通过其 **ONNX 运行时后端**（执行 ONNX 图形的通用引擎)加载和运行它，而不需要原始训练代码。

<a id="rocm-and-the-migraphx-execution-provider"></a>

### ROCm 和 MIGraphX 执行提供程序

**[ROCm](https://rocm.docs.amd.com/)** 是 AMD 用于 GPU 计算的开放软件平台。**[MIGraphX](https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/)** 是 AMD 的图形优化和 推理 引擎。当 ONNX 运行时与 **MIGraphX 执行提供程序**一起运行时，它将 ONNX 图形编译为 AMD GPU 的优化内核，从而在上实现快速 推理 **AMD Instinct** 硬件。

<a id="amd-instinct-gpus"></a>

### AMD Instinct GPU

**AMD Instinct** GPU（例如 MI300X、MI325X 和 MI355X）是 AMD 专为 AI 训练和 推理 设计的数据中心加速器。

<a id="perf_analyzer"></a>

### `perf_analyzer`

**`perf_analyzer`** 是 Triton 推理 服务器附带的标准基准测试客户端。它将受控的合成请求流发送到正在运行的服务器并报告 吞吐量 和 延迟，这使您可以测量自己硬件上的服务性能。

<details markdown="1">
<summary>查看英文原文 · 单元 2</summary>

## Key concepts

This section introduces the main technical concepts involved in this tutorial.

### Click-through rate (CTR) prediction

**CTR prediction** estimates the probability that a user will click on a particular item—an ad, a product, or a recommendation. It is one of the most economically important machine-learning problems, powering recommender systems and online advertising. The input is typically a mix of **numerical features** (for example, counts) and **categorical features** (for example, a device type or a user ID), and the output is a single probability between 0 and 1.

### FinalNet

**[FinalNet](https://arxiv.org/abs/2301.01624)** is a neural-network architecture for CTR prediction introduced at SIGIR 2023. It currently sits at the top of the [BARS Criteo\_x4 leaderboard](https://openbenchmark.github.io/BARS/CTR/leaderboard/criteo_x4.html), a community benchmark that ranks CTR models on a standardized dataset. It is used here as a realistic, production-relevant architecture to train and serve CTR models.

### The Criteo\_x4 dataset

**[Criteo\_x4](https://huggingface.co/datasets/reczoo/Criteo_x4)** is a large, publicly available advertising dataset widely used to benchmark CTR models. Each row describes one ad impression with 13 numerical and 26 categorical features, plus a label indicating whether the ad was clicked. It is publicly available and requires no access token to download.

### Triton Inference Server

**[Triton Inference Server](https://github.com/ROCm/triton-inference-server-server/tree/rocm7.2_r25.12)** is an open-source model-serving system. You hand it a trained model and it exposes HTTP/gRPC endpoints that clients call to run inference, while it handles batching, concurrency, and multiple model formats for you. This tutorial uses the **ROCm fork of Triton Inference Server**, which adds support for AMD GPUs.

### ONNX and the ONNX Runtime backend

**[ONNX (Open Neural Network Exchange)](https://onnx.ai/)** is a portable, framework-independent file format for trained models. Exporting our PyTorch model to ONNX allows Triton Inference Server to load and run it through its **ONNX Runtime backend**—a generic engine that executes ONNX graphs—without requiring the original training code.

### ROCm and the MIGraphX execution provider

**[ROCm](https://rocm.docs.amd.com/)** is AMD's open software platform for GPU computing. **[MIGraphX](https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/)** is AMD's graph-optimization and inference engine. When ONNX Runtime runs with the **MIGraphX execution provider**, it compiles the ONNX graph into optimized kernels for AMD GPUs, which enables fast inference on **AMD Instinct** hardware.

### AMD Instinct GPUs

**AMD Instinct** GPUs (such as the MI300X, MI325X, and MI355X) are AMD's data-center accelerators designed for AI training and inference.

### `perf_analyzer`

**`perf_analyzer`** is the standard benchmarking client that ships with Triton Inference Server. It sends a controlled stream of synthetic requests to a running server and reports throughput and latency, which lets you measure serving performance on your own hardware.

</details>

<!-- cell:3 -->

<a id="prerequisites"></a>

## 前置条件

<a id="hardware"></a>

### 硬件

- AMD Instinct GPU：本教程在 AMD Instinct **MI300X** 和 **MI325X** 上进行了测试。确保您使用的是 AMD Instinct GPU 或支持 ROCm 的兼容硬件，并且您的系统满足[官方要求](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)。

<a id="software"></a>

### 软件

下表列出了**本教程测试的确切版本**。其他版本可能也有效，但尚未经过验证。

<table>
  <thead>
    <tr><th>组件</th><th>版本（已测试）</th><th>参考</th></tr>
  </thead>
  <tbody>
    <tr>
      <td>ROCm</td>
      <td>7.2.0</td>
      <td><a href="https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html">快速入门安装指南</a></td>
    </tr>
    <tr>
      <td>Docker</td>
      <td>24.0.7</td>
      <td><a href="https://docs.docker.com/get-docker/">Docker安装指南</a></td>
    </tr>
    <tr>
      <td><code>rocm/primus:v26.2</code></td>
      <td>v26.2</td>
      <td>训练图像（拉入<a href="#step-2-train-finalnet-on-amd-gpus">Step 2</a>）</td>
    </tr>
    <tr>
      <td><code>rocm/tritoninferenceserver:tritoninferenceserver-25.12.<br>amd1_rocm7.2_ubuntu24.04_py3.12</code></td>
      <td>25.12 (rocm7.2)</td>
      <td>Triton 推理 服务器映像（拉入 <a href="#step-1-get-the-triton-inference-server-docker-image">Step 1</a>）</td>
    </tr>
    <tr>
      <td><code>nvcr.io/nvidia/tritonserver:25.12-py3-sdk</code></td>
      <td>25.12-py3-sdk</td>
      <td><a href="#step-5-measure-serving-performance-with-perf_analyzer">Step 5中使用的Benchmark客户端</a></td>
    </tr>
    <tr>
      <td>Python</td>
      <td>3.12.11</td>
      <td>用于结果绘图单元</td>
    </tr>
  </tbody>
</table>

您**不需要**需要手动克隆或构建任何组件 - 组件的图像由下面的单元命令自动拉取，并且 **Criteo\_x4** 数据集是公开可用的（无需登录或 API 令牌）。

> **提示：** 在开始之前，通过在终端中运行 `docker run hello-world` 来确认 Docker 已安装并且可以在没有 `sudo` 的情况下运行。

<details markdown="1">
<summary>查看英文原文 · 单元 3</summary>

## Prerequisites

### Hardware

- An AMD Instinct GPU: This tutorial was tested on an AMD Instinct **MI300X** and **MI325X**. Ensure you are using an AMD Instinct GPU or compatible hardware with ROCm support, and that your system meets the [official requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

### Software

The table below lists the **exact versions this tutorial was tested against**. Other versions may work, but they have not been validated.

<table>
  <thead>
    <tr><th>Component</th><th>Version (tested)</th><th>Reference</th></tr>
  </thead>
  <tbody>
    <tr>
      <td>ROCm</td>
      <td>7.2.0</td>
      <td><a href="https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html">Quick-start install guide</a></td>
    </tr>
    <tr>
      <td>Docker</td>
      <td>24.0.7</td>
      <td><a href="https://docs.docker.com/get-docker/">Docker install guide</a></td>
    </tr>
    <tr>
      <td><code>rocm/primus:v26.2</code></td>
      <td>v26.2</td>
      <td>Training image (pulled in <a href="#step-2-train-finalnet-on-amd-gpus">Step 2</a>)</td>
    </tr>
    <tr>
      <td><code>rocm/tritoninferenceserver:tritoninferenceserver-25.12.<br>amd1_rocm7.2_ubuntu24.04_py3.12</code></td>
      <td>25.12 (rocm7.2)</td>
      <td>Triton Inference Server image (pulled in <a href="#step-1-get-the-triton-inference-server-docker-image">Step 1</a>)</td>
    </tr>
    <tr>
      <td><code>nvcr.io/nvidia/tritonserver:25.12-py3-sdk</code></td>
      <td>25.12-py3-sdk</td>
      <td>Benchmark client used in <a href="#step-5-measure-serving-performance-with-perf_analyzer">Step 5</a></td>
    </tr>
    <tr>
      <td>Python</td>
      <td>3.12.11</td>
      <td>For the results-plotting cells</td>
    </tr>
  </tbody>
</table>

You do **not** need to clone or build any of the components manually—images of the components are pulled automatically by the cell commands below, and the **Criteo\_x4** dataset is publicly available (no login or API token required).

> **Tip:** Confirm Docker is installed and that it can be run without `sudo` by running `docker run hello-world` in a terminal before you start.

</details>

<!-- cell:4 -->

<a id="environment-setup"></a>

## 环境设置

在运行本教程的其余部分之前，请确保您的计算机已准备就绪并且可以在其上运行此Notebook。  

<a id="1-confirm-that-your-gpu-is-visible"></a>

### 1.确认您的GPU可见

ROCm 附带了 `rocm-smi`，这是一个命令行工具，列出了系统可以看到的 AMD GPU。运行下面的单元格以确认您的 AMD Instinct GPU 可见。如果未找到该命令或未出现 GPU，则需要修复 ROCm 安装或驱动程序，然后才能继续。

<details markdown="1">
<summary>查看英文原文 · 单元 4</summary>

## Environment setup

Before running the rest of the tutorial, make sure your machine is ready and that you can run this notebook on it.  

### 1. Confirm that your GPU is visible

ROCm ships `rocm-smi`, a command-line tool that lists the AMD GPUs the system can see. Run the cell below to confirm that your AMD Instinct GPU is visible. If the command is not found or no GPU appears, your ROCm installation or drivers need to be fixed before you continue.

</details>

```python
%%bash
# List the AMD GPUs visible to the host. You should see your Instinct GPU(s) here.
rocm-smi --showproductname
```

<!-- cell:6 -->

<a id="2-run-this-notebook-in-jupyterlab"></a>

### 2. 在 JupyterLab 中运行此Notebook

此Notebook可从 [AI Developer Hub GitHub 存储库](https://github.com/ROCm/gpuaidev) 获取。 

如果您已经在 J​​upyterLab 中阅读本文，请跳到下一步。否则，以下是如何在主机上启动此Notebook。

将 JupyterLab 安装到主机上的 Python 环境中：

   ```bash
   python3 -m pip install jupyterlab
   ```

从包含此Notebook的目录中，启动绑定到 `localhost` 的服务器：

   ```bash
   python3 -m jupyter lab --no-browser --ip=127.0.0.1 --port=8888 --notebook-dir=.
   ```

启动时，它会打印一个包含一次性令牌的 URL，格式为 `http://127.0.0.1:8888/lab?token=<long-token>`。

**如果 GPU 位于通过 SSH 访问的远程计算机上**，请在本地计算机上打开第二个终端并转发端口，以便您可以在本地浏览器中访问服务器：

   ```bash
   ssh -L 8888:localhost:8888 <user>@<gpu-host>
   ```

将打印的 `http://127.0.0.1:8888/lab?token=...` URL 粘贴到浏览器中，然后在 JupyterLab 中打开此Notebook。要在完成后停止服务器，请在运行服务器的终端中按 `Ctrl+C` 两次。

<details markdown="1">
<summary>查看英文原文 · 单元 6</summary>

### 2. Run this notebook in JupyterLab

This notebook is available at the [AI Developer Hub GitHub repository](https://github.com/ROCm/gpuaidev). 

If you are already reading this inside JupyterLab, skip to the next step. Otherwise, here is how to launch this notebook on your host machine.

Install JupyterLab into a Python environment on the host:

   ```bash
   python3 -m pip install jupyterlab
   ```

From the directory that contains this notebook, start the server bound to `localhost`:

   ```bash
   python3 -m jupyter lab --no-browser --ip=127.0.0.1 --port=8888 --notebook-dir=.
   ```

On startup it prints a URL containing a one-time token, in the format of `http://127.0.0.1:8888/lab?token=<long-token>`.

**If the GPU is on a remote machine** you accessed via SSH, open a second terminal on your local machine and forward the port so you can reach the server in your local browser:

   ```bash
   ssh -L 8888:localhost:8888 <user>@<gpu-host>
   ```

Paste the printed `http://127.0.0.1:8888/lab?token=...` URL into your browser, then open this notebook inside JupyterLab. To stop the server when you are done, press `Ctrl+C` twice in the terminal where it is running.

</details>

<!-- cell:7 -->

<a id="3-create-a-workspace-directory"></a>

### 3.创建工作区目录

所有克隆的存储库、模型工件和 MIGraphX 编译缓存均应位于名为 `WORKSPACE` 的单个主机目录下。下面的单元格将 `WORKSPACE` 设置为环境变量，以便后面的每个单元格都可以重用它来创建 Triton 推理 服务器期望的目录布局 - 这些是 `WORKSPACE` 下的目录：

- `model_repository/FinalNet_onnx/1/`：导出的 ONNX 模型将放置的位置。 Triton 推理 服务器需要每个模型文件夹内有一个编号版本子目录 (`1`)。
- `migraphx_cache/`：用​​于编译的GPU内核的持久缓存，因此服务器不需要在每次重新启动时重新编译模型。

请随意将 `WORKSPACE` 的路径更改为具有几 GB 可用空间的任何位置。

<details markdown="1">
<summary>查看英文原文 · 单元 7</summary>

### 3. Create a workspace directory

All cloned repositories, model artifacts, and the MIGraphX compilation cache are expected to be under a single host directory referred to as `WORKSPACE`. The cell below sets `WORKSPACE` as an environment variable so that every later cell can reuse it to create the directory layout that Triton Inference Server expects—these are the directories under `WORKSPACE`:

- `model_repository/FinalNet_onnx/1/`: Where the exported ONNX model will be placed. Triton Inference Server requires a numbered version sub-directory (`1`) inside each model folder.
- `migraphx_cache/`: A persistent cache for compiled GPU kernels, so the server does not need to recompile the model on every restart.

Feel free to change the path for `WORKSPACE` to any location that has a few GBs of free space.

</details>

```python
import os
from pathlib import Path

# Change this path if you prefer a different location.
WORKSPACE = Path.home() / "triton_finalnet_workspace"

# Export as an environment variable so the %%bash cells below can use ${WORKSPACE}.
os.environ["WORKSPACE"] = str(WORKSPACE)

(WORKSPACE / "model_repository" / "FinalNet_onnx" / "1").mkdir(parents=True, exist_ok=True)
(WORKSPACE / "migraphx_cache").mkdir(parents=True, exist_ok=True)

print(f"WORKSPACE = {WORKSPACE}")
```

<!-- cell:9 -->

<a id="step-1-get-the-triton-inference-server-docker-image"></a>

## 步骤 1：获取 Triton 推理 服务器 Docker 镜像

[Triton 推理 服务器的 ROCm 分支](https://github.com/ROCm/triton-inference-server-server/tree/rocm7.2_r25.12) 通过 MIGraphX 执行提供程序和 HIP 运行时添加了 AMD GPU 支持。本教程使用 **AMD 发布的预构建 Docker 映像**，因此无需手动构建。

本教程中使用的映像对应于 `rocm7.2_r25.12` 版本。运行下面的单元格来拉动它。首次运行时下载可能需要几分钟时间，具体取决于您的网络连接。

<details markdown="1">
<summary>查看英文原文 · 单元 9</summary>

## Step 1: Get the Triton Inference Server Docker image

The [ROCm fork of Triton Inference Server](https://github.com/ROCm/triton-inference-server-server/tree/rocm7.2_r25.12) adds AMD GPU support through the MIGraphX execution provider and the HIP runtime. This tutorial uses a **pre-built Docker image AMD publishes** so no manual building is required.

The image used in this tutorial corresponds to the `rocm7.2_r25.12` release. Run the cell below to pull it. The download may take several minutes on first run, depending on your network connection.

</details>

```python
%%bash
docker pull rocm/tritoninferenceserver:tritoninferenceserver-25.12.amd1_rocm7.2_ubuntu24.04_py3.12
echo "Triton Inference Server image pull complete."
```

<!-- cell:11 -->

通过运行以下命令确认图像现在存在于本地，该命令应列出该图像：

<details markdown="1">
<summary>查看英文原文 · 单元 11</summary>

Confirm the image is now present locally by running this command, which should list the image:

</details>

```python
%%bash
docker images | grep tritoninferenceserver
```

<!-- cell:13 -->

<a id="step-2-train-finalnet-on-amd-gpus"></a>

## 步骤 2：在 AMD GPU 上训练 FinalNet

> **如果您已有模型，请跳过此步骤。** 此训练步骤对于工作流程是可选的。如果您已经有经过训练的 FinalNet 检查点，请跳至**[步骤 3](#step-3-export-the-trained-model-to-onnx)** 将其导出。如果您已有 ONNX 模型，请跳至**[步骤 4](#step-4-deploy-finalnet-with-the-migraphx-execution-provider)** 进行部署。否则，请继续在此处创建模型（有关详细信息，请参阅上面的**[已经有经过训练的模型？](#already-have-a-trained-model)**）。

FinalNet 模型在 [Criteo\_x4](https://huggingface.co/datasets/reczoo/Criteo_x4) 数据集上进行训练。训练使用 [linsun12/FuxiCTR](https://github.com/linsun12/FuxiCTR) 分支（一个针对最新 PyTorch 版本进行修复的 CTR 训练框架），模型架构和超参数来自 [BARS 基准存储库](https://github.com/reczoo/BARS)。

所有训练命令都在 **`rocm/primus:v26.2` 容器**（即用型 PyTorch-on-ROCm 映像）内运行。该容器以分离模式运行，`WORKSPACE` 文件夹安装在其上，教程的后续步骤由 `docker exec` 命令驱动。

<a id="start-the-training-container"></a>

### 启动训练容器

首先，拉取训练镜像并启动一个名为 `train_CTR` 的长期容器：

<details markdown="1">
<summary>查看英文原文 · 单元 13</summary>

## Step 2: Train FinalNet on AMD GPUs

> **Skip this step if you already have a model.** This training step is optional for the workflow. If you already have a trained FinalNet checkpoint, skip to **[Step 3](#step-3-export-the-trained-model-to-onnx)** to export it. If you already have an ONNX model, skip to **[Step 4](#step-4-deploy-finalnet-with-the-migraphx-execution-provider)** to deploy it. Otherwise, continue here to create a model (see **[Already have a trained model?](#already-have-a-trained-model)** above for details).

The FinalNet model is trained on the [Criteo\_x4](https://huggingface.co/datasets/reczoo/Criteo_x4) dataset. The training uses the [linsun12/FuxiCTR](https://github.com/linsun12/FuxiCTR) fork (a CTR training framework with fixes for recent PyTorch versions) and the model architecture and hyperparameters come from the [BARS benchmark repository](https://github.com/reczoo/BARS).

All training commands run **inside a `rocm/primus:v26.2` container** (a ready-to-use PyTorch-on-ROCm image). The container is run in detached mode and the `WORKSPACE` folder is mounted onto it, with subsequent steps of the tutorial driven by the `docker exec` command.

### Start the training container

First, pull the training image and launch a long-lived container named `train_CTR`:

</details>

```python
%%bash
docker pull rocm/primus:v26.2

docker run \
    --name train_CTR \
    --device=/dev/kfd \
    --device=/dev/dri \
    -d --net=host \
    --ipc=host \
    -v ${WORKSPACE}:/workspace \
    rocm/primus:v26.2 \
    sleep infinity

echo "Container train_CTR started."
```

<!-- cell:15 -->

<a id="clone-the-training-repositories"></a>

### 克隆训练存储库

在容器内，克隆**BARS**基准配置、**FuxiCTR**训练框架和**Datasets**参考存储库，然后安装**FuxiCTR**及其依赖项：

<details markdown="1">
<summary>查看英文原文 · 单元 15</summary>

### Clone the training repositories

Inside the container, clone the **BARS** benchmark configs, the **FuxiCTR** training framework, and the **Datasets** reference repo, and then install **FuxiCTR** and its dependencies:

</details>

```python
%%bash
# Clone BARS benchmark configs, the FuxiCTR training fork, and the Datasets reference repo
docker exec train_CTR bash -c "
cd /workspace
git clone https://github.com/reczoo/BARS.git
git clone https://github.com/linsun12/FuxiCTR.git
cd FuxiCTR && pip install -r requirements.txt -q && pip install -e . -q
cd /workspace
git clone https://github.com/reczoo/Datasets.git
echo 'Repository setup complete.'
"
```

<!-- cell:17 -->

<a id="download-the-criteo_x4-dataset"></a>

### 下载 Criteo\_x4 数据集

下载公共 **Criteo\_x4** 数据集（压缩后约 4 GB）并将其解压到 **FuxiCTR** 期望的位置。不需要登录或令牌：

<details markdown="1">
<summary>查看英文原文 · 单元 17</summary>

### Download the Criteo\_x4 dataset

Download and unzip the public **Criteo\_x4** dataset (about 4 GB compressed) into the location **FuxiCTR** expects. No login or token is required:

</details>

```python
%%bash
# Download and extract the Criteo_x4 dataset (~4 GB compressed)
docker exec train_CTR bash -c "
mkdir -p /workspace/FuxiCTR/model_zoo/data/Criteo/Criteo_x4
cd /workspace/FuxiCTR/model_zoo/data/Criteo/Criteo_x4
wget -q -O Criteo_x4.zip \
    'https://huggingface.co/datasets/reczoo/Criteo_x4/resolve/main/Criteo_x4.zip?download=true'
apt-get update -q
apt-get install -y -q unzip
unzip -q Criteo_x4.zip
echo 'Dataset extracted:'
ls -lh *.csv
"
```

<!-- cell:19 -->

<a id="what-the-dataset-looks-like"></a>

#### 数据集是什么样的

**Criteo\_x4** 是一个表格数据集：每一行都是 **一次广告展示**，列是输入特征和要预测的标签。每行有：

- **`label`**：如果广告被点击，则为 `1`，否则为 `0`（这是模型预测的目标）。
- **13 个数字特征**，通常命名为 `I1`–`I13`：整数计数（当值缺失时，有些可能为空）。
- **26 个分类特征**，通常命名为 `C1`–`C26`：匿名 32 位哈希标识符（8 个字符的十六进制字符串），以避免暴露原始用户数据。

单行如下所示，此处显示为 `label`，后跟许多数字和分类列（在现实生活中，一行包含所有 13 + 26 列，以逗号或制表符分隔）：

```text
label, I1, I2, I3, ..., I13,  C1,        C2,        ..., C26
1,     5,  110, 0,  ..., 2,    68fd1e64,  80e26c9b,  ..., 3a171bb3
0,     ,   1,   14, ..., 1,    05db9164,  8947f767,  ..., 49d68486
```

<a id="using-your-own-data"></a>

#### 使用您自己的数据

FuxiCTR 是**配置驱动的**，因此您可以通过以下方式在不更改模型代码的情况下训练自己的 CTR 数据：

1. 将数据（训练/验证/测试）导出到 CSV 文件，每个文件每个功能一列，外加一个二进制 `label` 列。
2. 描述数据集配置中的这些列（`dataset_config.yaml` 条目），声明每列的名称和类型（`numeric` 表示连续值，`categorical` 表示 ID/字符串），以及 CSV 文件的路径。
3. 在下面的[启动训练](#launch-training)中，将`run_expid.py`指向您自己的数据的配置，而不是**Criteo\_x4**的配置。

有关确切的配置架构，请参阅 [FuxiCTR 文档](https://github.com/linsun12/FuxiCTR) 和 `model_zoo/` 下的示例 `dataset_config` 文件。随后的部署、ONNX 导出和基准测试步骤保持不变；只有 `config.pbtxt` 中的输入特征形状（**[步骤 4](#step-4-deploy-finalnet-with-the-migraphx-execution-provider)**）需要与您的特征计数相匹配。

<details markdown="1">
<summary>查看英文原文 · 单元 19</summary>

#### What the dataset looks like

**Criteo\_x4** is a tabular dataset: every row is **one ad impression**, and the columns are the input features and the label to be predicted. Each row has:

- **`label`**: `1` if the ad was clicked, `0` otherwise (this is the target the model predicts).
- **13 numerical features**, conventionally named `I1`–`I13`: Integer counts (some may be empty when a value is missing).
- **26 categorical features**, conventionally named `C1`–`C26`: Anonymized 32-bit hashed identifiers (8-character hex strings) to avoid exposing raw user data.

A single row looks like the following, shown here as `label` followed by a number of numerical and categorical columns (in real life a row has all the 13 + 26 columns, which are comma or tab-separated):

```text
label, I1, I2, I3, ..., I13,  C1,        C2,        ..., C26
1,     5,  110, 0,  ..., 2,    68fd1e64,  80e26c9b,  ..., 3a171bb3
0,     ,   1,   14, ..., 1,    05db9164,  8947f767,  ..., 49d68486
```

#### Using your own data

FuxiCTR is **configuration-driven**, so you can train on your own CTR data without changing the model code by:

1. Exporting your data (train/validation/test) to CSV files, each with one column per feature plus a binary `label` column.
2. Describing those columns in a dataset configuration (a `dataset_config.yaml` entry), declaring each column's name and type ( `numeric` for continuous values and `categorical` for IDs/strings), along with the paths to your CSV files.
3. In [Launch training](#launch-training) below, pointing `run_expid.py` to the configuration for your own data instead of the configuration for **Criteo\_x4**.

See the [FuxiCTR documentation](https://github.com/linsun12/FuxiCTR) and the sample `dataset_config` files under `model_zoo/` for the exact configuration schema. The deployment, ONNX export, and benchmarking steps that follow remain unchanged; only the input feature shapes in `config.pbtxt` (**[Step 4](#step-4-deploy-finalnet-with-the-migraphx-execution-provider)**) need to match your feature counts.

</details>

<!-- cell:20 -->

<a id="launch-training"></a>

### 启动培训

使用 BARS 基准配置开始训练 FinalNet。完整日志会传输到容器内的 `run.log`，因此您可以随时检查进度：

<details markdown="1">
<summary>查看英文原文 · 单元 20</summary>

### Launch training

Start training FinalNet using the BARS benchmark configuration. The full logs are streamed to `run.log` inside the container, so you can inspect the progress at any time:

</details>

```python
%%bash
# Train FinalNet on Criteo_x4 using the BARS benchmark configuration.
# Full logs are written to run.log inside the container.
docker exec train_CTR bash -c "
cd /workspace/FuxiCTR/model_zoo/FinalNet
python run_expid.py \
    --config /workspace/BARS/ranking/ctr/FinalNet/FinalNet_criteo_x4_001/FinalNet_criteo_x4_tuner_config_05 \
    --expid FinalNet_criteo_x4_001_041_449ccb21 \
    --gpu 0 2>&1 | tee run.log
"
```

<!-- cell:22 -->

训练完成后，FuxiCTR 将训练后的模型 检查点 写入容器内 `checkpoints/` 的**数据集特定子目录**。对于此配置，目录为 `model_zoo/FinalNet/checkpoints/criteo_x4_001_a5e05ce7/FinalNet_criteo_x4_001_041_449ccb21.model`，其中前缀 `criteo_x4_001_a5e05ce7` 是在 BARS 配置中声明的数据集 ID。 **[步骤 3](#step-3-export-the-trained-model-to-onnx)** 使用此 检查点 作为 ONNX 导出脚本的输入。您可以将您的运行与 `FinalNet_criteo_x4_001` 配置的公共 [BARS 排行榜](https://openbenchmark.github.io/BARS/CTR/leaderboard/criteo_x4.html) 条目进行比较。

> **注意：** 在完整的 Criteo\_x4 数据集上训练 CTR 模型是一项长期运行的工作，所需时间取决于您的硬件。您可以让训练在后台运行并继续探索本教程。

<details markdown="1">
<summary>查看英文原文 · 单元 22</summary>

When training finishes, FuxiCTR writes the trained model checkpoint into a **dataset-specific subdirectory** of `checkpoints/` inside the container. For this configuration, the directory is `model_zoo/FinalNet/checkpoints/criteo_x4_001_a5e05ce7/FinalNet_criteo_x4_001_041_449ccb21.model`, where the prefix `criteo_x4_001_a5e05ce7` is the dataset ID declared in the BARS configuration. **[Step 3](#step-3-export-the-trained-model-to-onnx)** uses this checkpoint as input to the ONNX export script. You can compare your run against the public [BARS leaderboard](https://openbenchmark.github.io/BARS/CTR/leaderboard/criteo_x4.html) entry for the `FinalNet_criteo_x4_001` configuration.

> **Note:** Training a CTR model on the full Criteo\_x4 dataset is a long-running job, and the time it takes depends on your hardware. You can let the training run in the background and continue to explore this tutorial.

</details>

<!-- cell:23 -->

<a id="step-3-export-the-trained-model-to-onnx"></a>

## 步骤 3：将训练好的模型导出到 ONNX

为了使用 Triton 推理 服务器为模型提供服务，将其从 PyTorch 检查点 转换为便携式 ONNX 格式。转换脚本 [`export_finalnet_to_onnx.py`](https://github.com/ROCm/triton-inference-server-server/blob/rocm7.2_r25.12/docs/perf_benchmark/FinalNet/export_finalnet_to_onnx.py) 读取 FuxiCTR 检查点 和配置，在 PyTorch 中重建模型，并将其跟踪到 ONNX 图（这是模型计算步骤的便携式表示）。

该脚本将结果直接写入 `WORKSPACE` 作为 `model_repository/FinalNet_onnx/1/model.onnx`，因此 Triton 推理 服务器可以加载它而无需任何进一步的重组。

该脚本被下载**到 FuxiCTR FinalNet 目录** (`model_zoo/FinalNet/`) 并从那里运行。在同一训练容器内运行导出：

<details markdown="1">
<summary>查看英文原文 · 单元 23</summary>

## Step 3: Export the trained model to ONNX

To serve the model with Triton Inference Server, it is converted from a PyTorch checkpoint into the portable ONNX format. The conversion script [`export_finalnet_to_onnx.py`](https://github.com/ROCm/triton-inference-server-server/blob/rocm7.2_r25.12/docs/perf_benchmark/FinalNet/export_finalnet_to_onnx.py) reads the FuxiCTR checkpoint and configuration, reconstructs the model in PyTorch, and traces it to an ONNX graph (which is a portable representation of the model's computation steps).

The script writes the result directly into the `WORKSPACE` as `model_repository/FinalNet_onnx/1/model.onnx`, so Triton Inference Server can load it without any further restructuring.

The script is downloaded **into the FuxiCTR FinalNet directory** (`model_zoo/FinalNet/`) and run from there. Run the export inside the same training container:

</details>

```python
%%bash
# Download the export script into the FuxiCTR FinalNet directory and run it from
# there, so its `from src import FinalNet` import resolves against model_zoo/FinalNet/src.
docker exec train_CTR bash -c "
wget -q -O /workspace/FuxiCTR/model_zoo/FinalNet/export_finalnet_to_onnx.py \
    https://raw.githubusercontent.com/ROCm/triton-inference-server-server/rocm7.2_r25.12/docs/perf_benchmark/FinalNet/export_finalnet_to_onnx.py

cd /workspace/FuxiCTR/model_zoo/FinalNet
python export_finalnet_to_onnx.py \
    --checkpoint /workspace/FuxiCTR/model_zoo/FinalNet/checkpoints/criteo_x4_001_a5e05ce7/FinalNet_criteo_x4_001_041_449ccb21.model \
    --config-dir /workspace/BARS/ranking/ctr/FinalNet/FinalNet_criteo_x4_001/FinalNet_criteo_x4_tuner_config_05 \
    --expid FinalNet_criteo_x4_001_041_449ccb21 \
    --output /workspace/model_repository/FinalNet_onnx/1/model.onnx
"
```

<!-- cell:25 -->

<a id="step-4-deploy-finalnet-with-the-migraphx-execution-provider"></a>

## 步骤 4：使用 MIGraphX 执行提供程序部署 FinalNet

> **注意：这是本教程的核心。** 如果您跳过了上述任何步骤，请确保您已运行 **[环境设置](#environment-setup)** 中的单元，并且您的 ONNX 模型位于 `${WORKSPACE}/model_repository/FinalNet_onnx/1/model.onnx` 中，然后再继续。如果您带来了自己的非 FinalNet 模型，请编辑下面的 `config.pbtxt` 文件中的输入/输出名称和形状以匹配您的模型。

存储库中的 ONNX 模型现已准备好提供服务。 Triton 推理 服务器通过其**ONNX 运行时后端**加载模型。模型的 `config.pbtxt` 文件中的 `execution_accelerators` 设置通过 **MIGraphX 执行提供程序**路由执行，该提供程序使用 [MIGraphX](https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/) 将图形编译为优化的 AMD GPU 内核。

<a id="model-configuration"></a>

### 模型配置

Triton 推理 服务器读取描述模型输入、输出、批处理行为和加速器的 `config.pbtxt` 文件。规范文件位于 ROCm 分支中的 [`docs/perf_benchmark/FinalNet/amd/config.pbtxt`](https://github.com/ROCm/triton-inference-server-server/blob/rocm7.2_r25.12/docs/perf_benchmark/FinalNet/amd/config.pbtxt)；它看起来像这样：

```protobuf
name: "FinalNet_onnx"
backend: "onnxruntime"
max_batch_size: 8192

input [
  { name: "numerical_features"  data_type: TYPE_FP32  dims: [ 13 ] },
  { name: "categorical_features" data_type: TYPE_INT64 dims: [ 26 ] }
]
output [
  { name: "output" data_type: TYPE_FP32 dims: [ 1 ] }
]

parameters {
  key: "execution_accelerators"
  value: {
    string_value: "{\"gpu_execution_accelerator\": [{\"name\": \"migraphx\"}]}"
  }
}

instance_group [ { count: 1  kind: KIND_GPU } ]

dynamic_batching {
  preferred_batch_size: [ 64, 128, 256, 512, 1024, 2048, 4096, 8192 ]
  max_queue_delay_microseconds: 100
}
```

有几点需要注意：

- `backend: "onnxruntime"` 选择 ONNX 运行时后端。
- `execution_accelerators` 参数启用 `migraphx` GPU 加速器。
- `dynamic_batching` 允许 Triton 推理 服务器将传入请求分组为更大的批次，以提高 GPU 利用率。

下一个单元会将此规范文件下载到位于 `${WORKSPACE}/model_repository/FinalNet_onnx/config.pbtxt` 的模型存储库中。

<details markdown="1">
<summary>查看英文原文 · 单元 25</summary>

## Step 4: Deploy FinalNet with the MIGraphX execution provider

> **Note: This is the core of the tutorial.** If you have skipped any step above, make sure you have run the cells in **[Environment setup](#environment-setup)** and your ONNX model is in place at `${WORKSPACE}/model_repository/FinalNet_onnx/1/model.onnx` before continuing. If you brought your own non-FinalNet model, edit the input/output names and shapes in the `config.pbtxt` file below to match your model.

Your ONNX model in the repository is now ready to be served. Triton Inference Server loads the model through its **ONNX Runtime backend**. The `execution_accelerators` setting in the model's `config.pbtxt` file routes execution through the **MIGraphX execution provider**, which compiles the graph into optimized AMD GPU kernels using [MIGraphX](https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/).

### Model configuration

Triton Inference Server reads a `config.pbtxt` file that describes the model's inputs, outputs, batching behavior, and accelerator. The canonical file lives at [`docs/perf_benchmark/FinalNet/amd/config.pbtxt`](https://github.com/ROCm/triton-inference-server-server/blob/rocm7.2_r25.12/docs/perf_benchmark/FinalNet/amd/config.pbtxt) in the ROCm fork; it looks like this:

```protobuf
name: "FinalNet_onnx"
backend: "onnxruntime"
max_batch_size: 8192

input [
  { name: "numerical_features"  data_type: TYPE_FP32  dims: [ 13 ] },
  { name: "categorical_features" data_type: TYPE_INT64 dims: [ 26 ] }
]
output [
  { name: "output" data_type: TYPE_FP32 dims: [ 1 ] }
]

parameters {
  key: "execution_accelerators"
  value: {
    string_value: "{\"gpu_execution_accelerator\": [{\"name\": \"migraphx\"}]}"
  }
}

instance_group [ { count: 1  kind: KIND_GPU } ]

dynamic_batching {
  preferred_batch_size: [ 64, 128, 256, 512, 1024, 2048, 4096, 8192 ]
  max_queue_delay_microseconds: 100
}
```

A few things to note:

- `backend: "onnxruntime"` selects the ONNX Runtime backend.
- The `execution_accelerators` parameter enables the `migraphx` GPU accelerator.
- `dynamic_batching` allows Triton Inference Server to group incoming requests into larger batches for better GPU utilization.

The next cell downloads this canonical file into your model repository at `${WORKSPACE}/model_repository/FinalNet_onnx/config.pbtxt`.

</details>

```python
%%bash
# Download the AMD config.pbtxt into the model repository
wget -q -O ${WORKSPACE}/model_repository/FinalNet_onnx/config.pbtxt \
    https://raw.githubusercontent.com/ROCm/triton-inference-server-server/rocm7.2_r25.12/docs/perf_benchmark/FinalNet/amd/config.pbtxt

# This single-GPU tutorial serves a single model instance. The upstream perf-benchmark
# config requests 32 instances on one GPU, which oversubscribes the device and makes
# perf_analyzer stall at higher concurrency; pin instance_group to a single instance.
sed -i -E 's/count:[[:space:]]*[0-9]+/count: 1/' ${WORKSPACE}/model_repository/FinalNet_onnx/config.pbtxt

echo "Model repository layout:"
find ${WORKSPACE}/model_repository -type f | sort
```

<!-- cell:27 -->

<a id="start-the-triton-inference-server"></a>

### 启动 Triton 推理 服务器

下一个单元启动 Triton 推理 服务器容器，安装模型存储库和 MIGraphX 缓存，并**轮询服务器的就绪端点**，直到准备就绪。第一次启动比后面的启动慢得多，因为 MIGraphX 执行提供程序必须将 ONNX 图编译到每个模型实例的 GPU 内核中，并将它们存储在已安装的缓存中。在冷启动时，这可能需要几分钟，而随后的重新启动将重用缓存，并且应该只需要几秒钟。

> **注意：** 如果重新运行此单元，请先使用 `docker rm -f tritonserver_container` 删除现有容器。

<details markdown="1">
<summary>查看英文原文 · 单元 27</summary>

### Start the Triton Inference Server

The next cell launches the Triton Inference Server container, mounts the model repository and the MIGraphX cache, and **polls the server's readiness endpoint** until it's ready. The first start is much slower than later ones because the MIGraphX execution provider has to compile the ONNX graph into GPU kernels for every model instance and store them in the mounted cache. On a cold start, this can take several minutes while subsequent restarts reuse the cache and should take only seconds.

> **Note:** If you re-run this cell, remove the existing container first with `docker rm -f tritonserver_container`.

</details>

```python
%%bash
# Start the Triton Inference Server container.
# The MIGraphX cache is mounted so compiled kernels persist across restarts.
TRITON_IMAGE="rocm/tritoninferenceserver:tritoninferenceserver-25.12.amd1_rocm7.2_ubuntu24.04_py3.12"

docker run \
    --name tritonserver_container \
    --device=/dev/kfd \
    --device=/dev/dri \
    --ipc=host \
    -d \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 8002:8002 \
    --net=host \
    -e ORT_MIGRAPHX_MODEL_CACHE_PATH=/migraphx_cache \
    -e ORT_MIGRAPHX_CACHE_PATH=/migraphx_cache \
    -v ${WORKSPACE}/model_repository:/models \
    -v ${WORKSPACE}/migraphx_cache:/migraphx_cache \
    "${TRITON_IMAGE}" \
    tritonserver --model-repository=/models --exit-on-error=false

# Poll the readiness endpoint instead of using a fixed sleep: a cold start
# compiles MIGraphX kernels for all model instances and can take several minutes.
echo "Waiting for the server to become ready (first start compiles MIGraphX kernels; this can take several minutes)..."
for i in $(seq 1 20); do
    code=$(curl -s -o /dev/null -w "%{http_code}" localhost:8000/v2/health/ready 2>/dev/null || true)
    if [ "$code" = "200" ]; then
        echo "Server ready after ~$((i * 20))s (HTTP 200)."
        break
    fi
    echo "  attempt ${i}/20: not ready yet (status: ${code:-no response}); waiting 20s..."
    sleep 20
done
docker logs tritonserver_container 2>&1 | tail -8
```

<!-- cell:29 -->

<a id="verify-the-server-is-ready"></a>

### 验证服务器是否准备就绪

Triton 推理 服务器在其 HTTP 端口 (8000) 上公开就绪端点。来自 `/v2/health/ready` 的 `HTTP 200` 响应意味着模型已加载并且服务器已准备好接受 推理 请求：

<details markdown="1">
<summary>查看英文原文 · 单元 29</summary>

### Verify the server is ready

Triton Inference Server exposes a readiness endpoint on its HTTP port (8000). An `HTTP 200` response from `/v2/health/ready` means the model is loaded and the server is ready to accept inference requests:

</details>

```python
%%bash
# A 200 response means Triton Inference Server is up and the model is ready.
curl -s -o /dev/null -w "HTTP %{http_code}\n" localhost:8000/v2/health/ready
```

<!-- cell:31 -->

<a id="step-5-measure-serving-performance-with-perf_analyzer"></a>

## 步骤 5：使用 `perf_analyzer` 衡量服务性能

服务器运行后，下一步是测量它在不同负载下的性能。这回答了一个实际问题：*使用此 GPU 上的此模型，服务器可以处理多少个请求以及多快？*这是通过在逐渐增加负载的同时测量两个量来回答的：

- **吞吐量**（每秒推理次数）：*服务器每单位时间完成多少次推理。*从**硬件角度**，它告诉您 GPU 的利用程度：当您发送更多并发请求时，吞吐量 不断攀升，直到 GPU 饱和且曲线变平。从**服务的角度**来看，它代表服务器的容量 - 例如，推荐服务每秒可以提供多少个预测，因此您需要多少个 GPU 来处理流量。
- **延迟**（每个请求的毫秒数，通常报告为 p50/p99 百分位数）：*单个请求等待其答案的时间。*从**硬件角度**，一旦 GPU 繁忙，延迟 就会随着请求一个接一个地排队而增加。从**服务角度**来看，它是您的最终用户（或上游服务）实际经历的延迟。 CTR 预测通常位于提供页面的关键路径上，因此 延迟 预算（例如，“p99 必须保持在 X 毫秒以下”）通常是一项硬性要求。

这两个数量之间需要权衡：推动 吞吐量 通常会增加 延迟。下面的扫描可让您找到**最大化 吞吐量 同时仍满足您的 延迟 预算**的工作点。这是使用 [`perf_analyzer`](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c%2B%2B/perf_analyzer/docs/cli.html) 在您自己的硬件上测量的，它以不断增加的并发级别发送受控的合成请求流，并为每个级别报告 吞吐量 和 延迟。

`perf_analyzer` 从官方 Triton 推理 服务器 SDK 容器映像 (`nvcr.io/nvidia/tritonserver:25.12-py3-sdk`) 运行，该映像附带与服务器兼容的 `perf_analyzer` 二进制文件。这里使用的标志是：

|旗帜|价值|意义|
|---|---|---|
| `-m FinalNet_onnx` | `FinalNet_onnx` |服务器上的 目标模型 名称 |
| `--input-data=random` |随机|用随机数据填充输入张量 |
| `-b 8192` | 8192 |根据要求 批大小 |
| `--concurrency-range 1:32:2` | 1 到 32，步骤 2 |横扫客户端并发级别 |
| `--measurement-interval 10000` | 10000 毫秒 |在 10 秒的窗口内测量每个并发级别以获得稳定的数字 |
| `-f /results/perf_results.csv` | `/results/perf_results.csv` |将每个并发结果写入 CSV 文件 |

> **关于扫描范围的注意事项：** 在 批大小 8192 上，GPU 在高并发之前就已经饱和，因此 吞吐量 曲线提前变平。扫描的并发上限为 32，并且每个级别使用 10 秒的测量窗口。将范围推得更高（例如，`1:72:2`）只会添加排队请求，从而停止运行而不会泄露新信息。如果你想探索更高的并发，请逐渐提高上限并保持更长的测量间隔。

`${WORKSPACE}` 作为 `/results` 安装到容器中，以便将 CSV 文件保存在主机上以便在下一步中进行绘图。

<details markdown="1">
<summary>查看英文原文 · 单元 31</summary>

## Step 5: Measure serving performance with `perf_analyzer`

With the server running, the next step is to measure how well it performs under different loads. This answers a practical question: *With this model on this GPU, how many requests can the server handle, and how quickly?* This is answered by measuring two quantities while gradually increasing the load:

- **Throughput** (inferences per second): *How many inferences the server completes per unit time.* From the **hardware perspective**, it tells you how fully the GPU is utilized: as you send more concurrent requests, throughput climbs until the GPU saturates and the curve flattens. From the **service perspective**, it represents the capacity of the server—for example, how many predictions per second a recommendation service can deliver, and therefore how many GPUs you would need to handle your traffic.
- **Latency** (milliseconds per request, usually reported as p50/p99 percentiles): *How long an individual request waits for its answer.* From the **hardware perspective**, latency rises as requests queue behind one another once the GPU is busy. From the **service perspective**, it is the delay your end user (or upstream service) actually experiences. CTR predictions typically sit on the critical path of serving a page, so a latency budget (e.g., "p99 must stay under X ms") is often a hard requirement.

There is a trade-off between the two quantities: pushing for throughput usually increases latency. The sweep below lets you find the operating point that **maximizes throughput while still meeting your latency budget**. This is measured **on your own hardware** using [`perf_analyzer`](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c%2B%2B/perf_analyzer/docs/cli.html), which sends a controlled stream of synthetic requests at increasing concurrency levels and reports throughput and latency for each level.

`perf_analyzer` is run from the official Triton Inference Server SDK container image (`nvcr.io/nvidia/tritonserver:25.12-py3-sdk`), which ships a `perf_analyzer` binary compatible with the server. The flags used here are:

| Flag | Value | Meaning |
|---|---|---|
| `-m FinalNet_onnx` | `FinalNet_onnx` | Target model name on the server |
| `--input-data=random` | random | Fill input tensors with random data |
| `-b 8192` | 8192 | Batch size per request |
| `--concurrency-range 1:32:2` | 1 to 32, step 2 | Sweep across client concurrency levels |
| `--measurement-interval 10000` | 10000 ms | Measure each concurrency level over a 10 s window for stable numbers |
| `-f /results/perf_results.csv` | `/results/perf_results.csv` | Write the per-concurrency results to a CSV file |

> **Note on the sweep range:** At batch size 8192, the GPU already saturates well before high concurrency, so the throughput curve flattens early. The sweep is capped at concurrency 32, and a 10-second measurement window per level is used. Pushing the range much higher (e.g., `1:72:2`) will only add queued requests that stall the run without revealing new information. If you want to explore higher concurrency, raise the upper bound gradually and keep the longer measurement interval.

`${WORKSPACE}` is mounted into the container as `/results` for the CSV file to be saved on the host for plotting in the next step.

</details>

```python
%%bash
# Note: no -it flag — a notebook cell has no interactive TTY, so -it would fail here.
docker run --rm --net=host \
    -v ${WORKSPACE}:/results \
    nvcr.io/nvidia/tritonserver:25.12-py3-sdk \
    perf_analyzer \
        -m FinalNet_onnx \
        --input-data=random \
        -b 8192 \
        --concurrency-range 1:32:2 \
        --measurement-interval 10000 \
        -f /results/perf_results.csv
```

<!-- cell:33 -->

<a id="how-to-read-the-output"></a>

### 如何读取输出

`perf_analyzer` 打印统计数据块以及每个并发级别的汇总表。对于每个级别，您将看到：

- **推理次数/秒**：吞吐量，即服务器每秒完成多少次推理。乘以 批大小 以转换为样本级别 吞吐量。
- **延迟 百分位数**（p50、p90、p99）：请求花费了多长时间。 p99 值是尾部 延迟，即 99% 的请求在其中完成。

一般来说，吞吐量 随着并发量的增加而增加，直到 GPU 饱和，而 延迟 随着请求开始排队而增加。正确的工作点取决于您的 延迟 目标。所有结果都写入`perf_results.csv`以供下一步分析。

<details markdown="1">
<summary>查看英文原文 · 单元 33</summary>

### How to read the output

`perf_analyzer` prints a statistics block plus a summary table per concurrency level. For each level you will see:

- **Inferences/Second**: Throughput, i.e., how many inferences the server completes per second. Multiply by the batch size to convert to sample-level throughput.
- **Latency percentiles** (p50, p90, p99): How long requests took. The p99 value is the tail latency, i.e. 99% of requests complete within it.

In general, throughput rises with concurrency until the GPU saturates, while latency grows as requests begin to queue. The right operating point depends on your latency target. All results are written to `perf_results.csv` for analysis in the next step.

</details>

<!-- cell:34 -->

<a id="step-6-plot-your-own-results"></a>

## 第 6 步：绘制自己的结果

`perf_analyzer` 在上一步中将 `perf_results.csv` 文件写入到您的 `WORKSPACE` 中。下面的单元格加载**您的** CSV 文件，并根据客户端并发情况绘制 吞吐量 和尾部 延迟，以便您可以直观地看到服务器在您的硬件上的行为方式。

该Notebook没有预先记录的数字，绘图完全由您刚刚生成的 CSV 文件生成。

<details markdown="1">
<summary>查看英文原文 · 单元 34</summary>

## Step 6: Plot your own results

`perf_analyzer` wrote a `perf_results.csv` file to your `WORKSPACE` in the previous step. The cells below load **your** CSV file and plot throughput and tail latency against client concurrency, so you can visualize how the server behaves on your hardware.

This notebook ships with no pre-recorded numbers, the plots are generated entirely from the CSV file you just produced.

</details>

```python
try:
    import matplotlib
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "-q"])
print("matplotlib ready.")
```

<!-- cell:36 -->

下一个单元格加载 CSV 文件并使用 `matplotlib` 绘制曲线。如果未找到 CSV 文件，单元格将打印提示而不是失败 — 运行 [步骤 5](#step-5-measure-serving-performance-with-perf_analyzer) 来生成它。

<details markdown="1">
<summary>查看英文原文 · 单元 36</summary>

The next cell loads the CSV file and plots the curves with `matplotlib`. If the CSV file is not found, the cell prints a hint instead of failing—run [Step 5](#step-5-measure-serving-performance-with-perf_analyzer) to generate it.

</details>

```python
import csv
import os
from pathlib import Path
import matplotlib.pyplot as plt

AMD_RED, AMD_ORANGE = "#ED1C24", "#F58220"
csv_path = Path(os.environ["WORKSPACE"]) / "perf_results.csv"

if not csv_path.exists():
    print(f"No results file at {csv_path}.")
    print("Run the perf_analyzer cell in Step 5 to generate it, then re-run this cell.")
else:
    concurrency, throughput, p99_ms = [], [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            concurrency.append(int(float(row["Concurrency"])))
            throughput.append(float(row["Inferences/Second"]))
            p99_ms.append(float(row["p99 latency"]) / 1000.0)  # microseconds -> ms

    order = sorted(range(len(concurrency)), key=lambda i: concurrency[i])
    concurrency = [concurrency[i] for i in order]
    throughput = [throughput[i] for i in order]
    p99_ms = [p99_ms[i] for i in order]
    print(f"Loaded {len(concurrency)} concurrency levels from {csv_path.name}.")
```

<!-- cell:38 -->

最后，根据上面加载的数据渲染 吞吐量 和 延迟 曲线：

<details markdown="1">
<summary>查看英文原文 · 单元 38</summary>

Finally, render the throughput and latency curves from the data loaded above:

</details>

```python
if csv_path.exists():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(concurrency, throughput, marker="o", color=AMD_RED)
    axes[0].set(xlabel="Client concurrency", ylabel="Throughput (inferences/sec)",
                title="FinalNet throughput on your hardware")

    axes[1].plot(concurrency, p99_ms, marker="s", color=AMD_ORANGE)
    axes[1].set(xlabel="Client concurrency", ylabel="p99 latency (ms)",
                title="FinalNet tail latency on your hardware")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.45)

    fig.tight_layout()
    plt.show()
else:
    print("Run Step 5 and the load cell above first.")
```

<!-- cell:40 -->

<a id="clean-up"></a>

### 清理

完成后，停止并删除本教程创建的容器以释放 GPU 内存和磁盘空间：

<details markdown="1">
<summary>查看英文原文 · 单元 40</summary>

### Clean up

When you are finished, stop and remove the containers created by this tutorial to free up GPU memory and disk space:

</details>

```python
%%bash
# Stop and remove the tutorial containers (ignores errors if they are already gone).
docker rm -f train_CTR tritonserver_container 2>/dev/null || true
echo "Cleanup complete."
```

<!-- cell:42 -->

<a id="summary"></a>

## 总结

在本教程中，您使用 AMD Instinct 硬件上的 GPU 加速 推理 服务处理了从训练到部署和基准测试的整个点击率预测模型。你有：  

- **在 ROCm PyTorch 容器内的公共 [Criteo\_x4](https://huggingface.co/datasets/reczoo/Criteo_x4) 数据集上使用 [FuxiCTR](https://github.com/linsun12/FuxiCTR) 训练** FinalNet。
- **将经过训练的 检查点 导出为便携式 ONNX 格式。
- 使用由 MIGraphX 执行提供程序加速的 ONNX 运行时后端在 [Triton 推理 服务器](https://github.com/ROCm/triton-inference-server-server/tree/rocm7.2_r25.12) 上**部署**模型。
- **使用 `perf_analyzer` 在您自己的硬件上测量** 吞吐量 和 延迟，并绘制结果。

从这里，您可以尝试提供不同的模型，试验 `config.pbtxt` 中的 `dynamic_batching` 设置，或者通过其 HTTP/gRPC API 将服务器集成到应用程序中。

<a id="further-reading"></a>

### 进一步阅读
                                                                                 
  - [配套博客：使用 Triton 推理 服务器（ONNX 运行时后端）提供 CTR 模型](https://rocm.blogs.amd.com/artificial-intelligence/triton-inference-server/README.html)                                                   
  - [ROCm人工智能开发者中心](https://rocm.docs.amd.com/projects/ai-developer-hub)
  - [MIGraphX 文档](https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/)     
  - [ROCm ONNX 运行时 + MIGraphX EP](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/rocm_ai/onnxruntime.html)                                      
  - [Triton 推理 服务器（ROCm 分支）](https://github.com/ROCm/triton-inference-server-server/tree/rocm7.2_r25.12)                                         
  - [Triton 推理 服务器 `perf_analyzer` CLI 参考](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/docs/cli.html)
  - [BARS点击率排行榜](https://openbenchmark.github.io/BARS/CTR/leaderboard/criteo_x4.html)                                                               
  - [FinalNet 论文 (SIGIR 2023)](https://arxiv.org/abs/2301.01624)
  - [Criteo\_x4 数据集](https://huggingface.co/datasets/reczoo/Criteo_x4)

<details markdown="1">
<summary>查看英文原文 · 单元 42</summary>

## Summary

In this tutorial, you handled a CTR prediction model all the way from training to deployment and benchmarking, using a GPU-accelerated inference service on AMD Instinct hardware. You have:  

- **Trained** FinalNet with [FuxiCTR](https://github.com/linsun12/FuxiCTR) on the public [Criteo\_x4](https://huggingface.co/datasets/reczoo/Criteo_x4) dataset inside a ROCm PyTorch container.
- **Exported** the trained checkpoint to the portable ONNX format.
- **Deployed** the model on [Triton Inference Server](https://github.com/ROCm/triton-inference-server-server/tree/rocm7.2_r25.12) using the ONNX Runtime backend accelerated by the MIGraphX execution provider.
- **Measured** throughput and latency on your own hardware with `perf_analyzer`, and plotted the results.

From here, you could try serving a different model, experimenting with the `dynamic_batching` settings in `config.pbtxt`, or integrating the server into an application through its HTTP/gRPC API.

### Further reading
                                                                                 
  - [Companion blog: Serving CTR models with Triton Inference Server (ONNX Runtime backend)](https://rocm.blogs.amd.com/artificial-intelligence/triton-inference-server/README.html)                                                   
  - [ROCm AI Developer Hub](https://rocm.docs.amd.com/projects/ai-developer-hub)
  - [MIGraphX documentation](https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/)     
  - [ROCm ONNX Runtime + MIGraphX EP](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/rocm_ai/onnxruntime.html)                                      
  - [Triton Inference Server (ROCm fork)](https://github.com/ROCm/triton-inference-server-server/tree/rocm7.2_r25.12)                                         
  - [Triton Inference Server `perf_analyzer` CLI reference](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/docs/cli.html)
  - [BARS CTR leaderboard](https://openbenchmark.github.io/BARS/CTR/leaderboard/criteo_x4.html)                                                               
  - [FinalNet paper (SIGIR 2023)](https://arxiv.org/abs/2301.01624)
  - [Criteo\_x4 dataset](https://huggingface.co/datasets/reczoo/Criteo_x4)

</details>
