> 原文作者：Wei Cai  
> [AMD 官方教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/fine_tune/slime_qwen3_4B_GRPO.html) · [固定版本源码](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/fine_tune/slime_qwen3_4B_GRPO.ipynb)  
> 上游版本：`0071bfd4bd3bd9525405c09a6ae69c983b09312a` · 同步日期：2026-10-12  
> 翻译状态：机器初译，待逐篇人工校对；代码与命令保持原文，未在 AMD GPU 上运行验证。

<!-- cell:1 -->

<a id="rl-training-with-slime-on-amd-gpus"></a>

# 使用 slime 进行 GRPO 训练
**作者**：Wei Cai
**知识水平**：中级

现代大型语言模型在 预训练 之后并没有停止改进。为了对现实世界的任务有用、一致和稳健，他们必须从反馈中学习。这就是 强化学习 (RL) 的用武之地。

本教程将引导您完成 [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) 大语言模型 的真实生产型 RL 训练管道，完全在具有 ROCm 的 AMD GPU 上运行。该工作流程由 [slime](https://github.com/THUDM/slime) 提供支持，这是一个专门为 LLM 规模的 RL 扩展而构建的 SGLang 原生后训练框架。

<a id="why-use-slime"></a>

## 为什么要使用史莱姆？

使用强化学习培训法学硕士具有挑战性，主要原因有两个：

- **推出一代价格昂贵**：
   您需要一个快速的 推理 引擎来对大量模型响应进行采样。

- **政策优化重磅**：
   您需要一个可跨 GPU 扩展的高度优化的训练堆栈。

[slime](https://github.com/THUDM/slime) 通过干净地分离和有效连接这两个组件来解决这两个挑战：

- **SGLang**：适用于高 吞吐量 部署一代
- **Megatron-LM**：用于分布式策略训练

它们共同构成了一个可扩展的模块化强化学习系统，与现代法学硕士工作负载保持一致。

<a id="what-is-grpo"></a>

## 什么是GRPO？

本教程使用 GRPO (组相对策略优化)，这是一种专为可扩展的 LLM 训练而设计的现代 RL 算法。

传统的强化学习方法通常依赖于单个参考基线，也称为批评模型，它可以是：

- 训练费用昂贵
- 难以稳定
- 对奖励噪音敏感

GRPO 采用了不同的方法。 GRPO 不是孤立地评估响应，而是：

- 对同一提示的多个响应进行采样
- 将它们分组在一起
- 计算群体内的相对优势

<a id="why-this-matters-for-llms"></a>

### 为什么这对法学硕士很重要

GRPO 具有多项实际优势：
- 无需单独的价值网络
- 更稳定的训练信号
- 更好的大批量部署的扩展行为
- 自然适合基于服务器的部署生成（SGLang）

这使得 GRPO 特别适合：
- 指令调整
- 推理能力提升
- 基于偏好的优化
- 多 GPU 系统上的大规模强化学习

<a id="what-youll-learn-in-this-notebook"></a>

## 您将在本Notebook中学到什么

学完本教程后，您将能够：

- 在 AMD GPU 上为 slime 设置启用 ROCm 的 Docker 环境
- 为 Qwen3-4B 配置 GRPO，包括推出和奖励设置
- 运行端到端 RL 训练循环，结合：
  - SGLang 用于生成
  - 用于优化的 Megatron-LM
- 了解可扩展的 LLM RL 培训背后的系统级设计选择

无论您是尝试训练后研究还是构建生产级 RL 管道，此Notebook都旨在为您提供工作代码和清晰的思维模型。让我们开始吧。

<details markdown="1">
<summary>查看英文原文 · 单元 1</summary>

# RL training with slime on AMD GPUs

**Author**: Wei Cai  
**Knowledge level**: Intermediate

Modern large language models don't stop improving after pretraining. To be useful, aligned, and robust for real-world tasks, they must learn from feedback. That's where Reinforcement Learning (RL) comes in.

This tutorial walks you through a real, production-style RL training pipeline for the [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) large language model, running entirely on AMD GPUs with ROCm. The workflow is powered by [slime](https://github.com/THUDM/slime), an SGLang-native post-training framework built specifically for RL scaling at LLM scale.

## Why use slime?

Training LLMs with RL is challenging for two key reasons:

- **Rollout generation is expensive**:
   You need a fast inference engine to sample large volumes of model responses.

- **Policy optimization is heavy**:
   You need a highly optimized training stack that scales across GPUs.

[slime](https://github.com/THUDM/slime) addresses both challenges by cleanly separating and efficiently connecting these two components:

- **SGLang**: For high-throughput rollout generation
- **Megatron-LM**: For distributed policy training

Together, they form a scalable, modular RL system that aligns with modern LLM workloads.

## What is GRPO?

This tutorial uses GRPO (Group Relative Policy Optimization), a modern RL algorithm designed for scalable LLM training.

Traditional RL methods often rely on a single reference baseline, also known as a critic model, which can be:

- Expensive to train
- Hard to stabilize
- Sensitive to reward noise

GRPO takes a different approach. Instead of evaluating a response in isolation, GRPO:

- Samples multiple responses for the same prompt
- Groups them together
- Computes relative advantages within the group

### Why this matters for LLMs

GRPO offers several practical advantages:
- No separate value network required
- More stable training signals
- Better scaling behavior for large batch rollouts
- Naturally fits server-based rollout generation (SGLang)

This makes GRPO especially well-suited for:
- Instruction tuning
- Reasoning improvement
- Preference-based optimization
- Large-scale RL on multi-GPU systems

## What you'll learn in this notebook

By the end of this tutorial, you'll be able to:

- Set up a ROCm-enabled Docker environment for slime on AMD GPUs
- Configure GRPO for Qwen3-4B, including rollout and reward settings
- Run an end-to-end RL training loop, combining:
  - SGLang for generation
  - Megatron-LM for optimization
- Understand the system-level design choices behind scalable LLM RL training

Whether you're experimenting with post-training research or building production-grade RL pipelines, this notebook is designed to give you both working code and clear mental models. Let's get started.

</details>

<!-- cell:2 -->

<a id="prerequisites"></a>

## 前置条件

本教程是使用以下设置进行开发和测试的。 

<a id="operating-system"></a>

### 操作系统

* **Ubuntu 22.04**：确保您的系统运行的是 Ubuntu 22.04。

<a id="hardware"></a>

### 硬件

* **AMD Instinct™ GPU**：本教程在八个 AMD Instinct MI300X GPU 的完整节点上进行了测试。确保您使用的是 AMD Instinct GPU 或支持 ROCm 的兼容硬件，并且您的系统满足[官方要求](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)。

<a id="software"></a>

### 软件

* **ROCm 7.0.0**：按照 [ROCm 安装指南](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html) 安装并验证 ROCm。安装后，使用以下命令确认您的设置：

    ``` bash
    amd-smi
    ```

    此命令列出了您的 AMD GPU 以及相关详细信息。

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

* **Ubuntu 22.04**: Ensure your system is running Ubuntu 22.04.

### Hardware

* **AMD Instinct™ GPUs**: This tutorial was tested on a full node of eight AMD Instinct MI300X GPUs. Ensure you are using AMD Instinct GPUs or compatible hardware with ROCm support and that your system meets the [official requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

### Software

* **ROCm 7.0.0**: Install and verify ROCm by following the [ROCm install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html). After installation, confirm your setup using:

    ``` bash
    amd-smi
    ```

    This command lists your AMD GPUs with relevant details.

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

<a id="system-validation"></a>

## 系统验证

在运行 AI 工作负载之前，务必确保您的 AMD 硬件配置正确且性能最佳。

一般来说，禁用 NUMA（非统一内存访问）自动平衡可以提高应用程序性能。但是，此设置可能会损害某些类型的工作负载的性能。

运行以下命令来验证当前的 NUMA 设置：

``` bash
cat /proc/sys/kernel/numa_balancing
``` 

输出 `0` 表示 NUMA 自动平衡已禁用。如果没有任何输出或者输出为`1`，请运行以下命令禁用NUMA自动平衡。

``` bash
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
``` 

有关详细信息，请参阅[禁用 NUMA 自动平衡](https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/system-optimization/mi300x.html#disable-numa-auto-balancing)。

<details markdown="1">
<summary>查看英文原文 · 单元 3</summary>

## System validation

Before running AI workloads, it's important to ensure that your AMD hardware is configured correctly and performing optimally.

Generally, application performance can benefit from disabling NUMA (Non-Uniform Memory Access) auto-balancing. However, this setting might be detrimental to performance with certain types of workloads.

Run this command to verify the current NUMA settings:

``` bash
cat /proc/sys/kernel/numa_balancing
``` 

An output of `0` indicates NUMA auto-balancing is disabled. If there is no output or the output is `1`, run the following command to disable NUMA auto-balancing.

``` bash
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
``` 

For more information, see [Disable NUMA auto-balancing](https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/system-optimization/mi300x.html#disable-numa-auto-balancing).

</details>

<!-- cell:4 -->

<a id="set-up-the-environment"></a>

## 设置环境

请按照以下步骤准备培训环境。

<a id="1-pull-the-docker-image"></a>

### 1. 拉取 Docker 镜像

确保您的系统满足[系统要求](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)。

拉取本教程所需的 Docker 映像：

``` bash
docker pull rlsys/slime:latest
```

<a id="2-launch-the-docker-container"></a>

### 2.启动Docker容器

启动 Docker 容器并映射必要的目录。 

``` bash
docker run -it \
  --device /dev/dri \
  --device /dev/kfd \
  -p 8265:8265 \
  --group-add video \
  --network host --ipc host \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --privileged \
  -v $HOME/.ssh:/root/.ssh \
  -v $HOME:$HOME \
  -w /workspace/notebooks \
  --shm-size 128G \
  --name slime \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  rlsys/slime:latest \
  /bin/bash
```

**注意**：如果退出后需要返回到 `slime` 容器，请使用以下命令：

``` bash
docker start slime
docker exec -it slime bash
```

**注意**：确保Notebook文件在启动后复制到 `/workspace` 目录或上传到 Jupyter Notebook 环境中。保存终端输出中提供的令牌或 URL，以便从 Web 浏览器访问Notebook。您可以从 [AI Developer Hub GitHub 存储库](https://github.com/ROCm/gpuaidev) 下载此Notebook。


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
在开始 RL 训练之前，您需要安装 slime——连接的核心框架
基于 SGLang 的部署生成以及基于 Megatron-LM 的策略优化。

由于 slime 正在积极开发中，持续的提交可能会带来行为变化
影响推出语义、奖励计算或训练稳定性。
为了确保本教程的可重复性和稳定性，请将安装固定到已知的
工作承诺。

在启用 ROCm 的 Docker 容器内运行以下命令：

<details markdown="1">
<summary>查看英文原文 · 单元 4</summary>

## Set up the environment

Follow these steps to prepare the training environment.

### 1. Pull the Docker image

Ensure your system meets the [System Requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

Pull the Docker image required for this tutorial:

``` bash
docker pull rlsys/slime:latest
```

### 2. Launch the Docker container

Launch the Docker container and map the necessary directories. 

``` bash
docker run -it \
  --device /dev/dri \
  --device /dev/kfd \
  -p 8265:8265 \
  --group-add video \
  --network host --ipc host \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --privileged \
  -v $HOME/.ssh:/root/.ssh \
  -v $HOME:$HOME \
  -w /workspace/notebooks \
  --shm-size 128G \
  --name slime \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  rlsys/slime:latest \
  /bin/bash
```

**Note**: If you need to return to the `slime` container after exiting it, use these commands:

``` bash
docker start slime
docker exec -it slime bash
```

**Note**: Ensure the notebook file is either copied to the `/workspace` directory or uploaded into the Jupyter Notebook environment after it starts. Save the token or URL provided in the terminal output to access the notebook from your web browser. You can download this notebook from the [AI Developer Hub GitHub repository](https://github.com/ROCm/gpuaidev).


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

### 4. Install the required libraries
Before starting RL training, you need to install slime — the core framework that connects
SGLang-based rollout generation with Megatron-LM-based policy optimization.

Since slime is under active development, ongoing commits could introduce behavior changes
that affect rollout semantics, reward computation, or training stability.
To ensure this tutorial is reproducible and stable, pin the installation to a known
working commit.

Run the following commands inside the ROCm-enabled Docker container:

</details>

```python
!git clone https://github.com/THUDM/slime.git
%cd slime
# Note -- You can run the latest upstream version. If you want a stable version, check out the following commit ID
!git checkout 0934a0e
# Install the package
!pip install -e .
```

<!-- cell:6 -->

在继续之前，请确认 slime 已正确安装并且对 Python 可见：

<details markdown="1">
<summary>查看英文原文 · 单元 6</summary>

Before continuing, confirm that slime is correctly installed and visible to Python:

</details>

```python
# Verify the installation and version of the required libraries
!pip list | grep slime
```

<!-- cell:8 -->

如果输出中出现 `slime`，则您的环境已准备好进行下一步。

<details markdown="1">
<summary>查看英文原文 · 单元 8</summary>

If `slime` appears in the output, your environment is ready for the next step.

</details>

<!-- cell:9 -->

<a id="run-grpo-training"></a>

## 运行 GRPO 训练

本部分将引导您完成设置和运行的端到端过程
组相对策略优化 (GRPO) Qwen3-4B 培训。

在较高层面上，GRPO 培训需要：
- 基础预训练模型（您想要改进的策略）
- 用于生成部署和计算奖励的训练数据集
- 保留的评估数据集，用于跟踪训练期间的泛化情况

<a id="1-download-the-model-and-datasets"></a>

### 1.下载模型和数据集

首先，下载基础Qwen3-4B模型，作为初始策略
对于 RL 微调。

对于训练，您将使用 `dapo-math-17k`，这是一个旨在评估
逐步的数学推理。这是一项需要进行相对比较的任务
多个模型输出之间的比较特别有效。

对于评估，请使用 `aime-2024`，它提供了一个干净的基准
监控推理性能而不泄露训练数据。

运行以下命令来下载所有必需的工件。首先，从 Hugging Face 下载基础 Qwen3-4B 型号 检查点：

<details markdown="1">
<summary>查看英文原文 · 单元 9</summary>

## Run GRPO training

This section walks you through the end-to-end process of setting up and running
Group Relative Policy Optimization (GRPO) training for Qwen3-4B.

At a high level, GRPO training requires:
- A base pretrained model (the policy you want to improve)
- A training dataset to use to generate rollouts and compute rewards
- A held-out evaluation dataset to track generalization during training

### 1. Download the model and datasets

First, download the base Qwen3-4B model, which serves as the initial policy
for RL fine-tuning.

For training, you will use `dapo-math-17k`, a dataset designed to evaluate
step-by-step mathematical reasoning. This is a task for which relative comparisons
between multiple model outputs are especially effective.

For evaluation, use `aime-2024`, which provides a clean benchmark to
monitor reasoning performance without leaking training data.

Run the following commands to download all required artifacts. First, download the base Qwen3-4B model checkpoint from Hugging Face:

</details>

```python
!hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B
```

<!-- cell:11 -->

接下来，下载用于数学推理任务的训练数据集（`dapo-math-17k`）：

<details markdown="1">
<summary>查看英文原文 · 单元 11</summary>

Next, download the training dataset (`dapo-math-17k`) for mathematical reasoning tasks:

</details>

```python
!hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k
```

<!-- cell:13 -->

然后下载 AIME 2024 评估数据集进行基准测试：

<details markdown="1">
<summary>查看英文原文 · 单元 13</summary>

Then download the AIME 2024 evaluation dataset for benchmarking:

</details>

```python
!hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /root/aime-2024
```

<!-- cell:15 -->

<a id="2-convert-the-checkpoint-format"></a>

### 2.转换检查点格式

在开始 GRPO 训练之前，请将预训练的 Hugging Face 检查点 转换为 Megatron Core 分布式格式。

需要进行此转换，因为 slime 使用 Megatron-LM 进行训练。 Megatron-LM 期望根据目标并行化策略（即 张量并行 和 流水线并行）布置模型权重。相比之下，Hugging Face 检查点以与框架无关的单进程格式存储权重。

这是针对给定模型和并行配置的一次性预处理步骤。你不需要每次训练都重复这个过程，只要
因为并行度设置保持不变。

运行以下命令来执行转换：

<details markdown="1">
<summary>查看英文原文 · 单元 15</summary>

### 2. Convert the checkpoint format

Before you can start GRPO training, convert the pretrained Hugging Face checkpoint into the Megatron Core distributed format.

This conversion is required because slime uses Megatron-LM for training. Megatron-LM expects model weights to be laid out according to the target parallelization strategy (that is, tensor parallelism and pipeline parallelism). Hugging Face checkpoints, by contrast, store weights in a framework-agnostic, single-process format.

This is a one-time preprocessing step for a given model and parallel configuration. You don't need to repeat it for every training run, as long
as the parallelism settings remain unchanged.

Run the following commands to perform the conversion:

</details>

```python
%%bash
# Navigate to the slime repository
cd /workspace/notebooks/slime

# Load model configuration arguments
source scripts/models/qwen3-4B.sh

# Locate megatron core installation path
MEGATRON_LM_PATH=$(pip list | grep megatron-core | awk '{print $NF}')

# Run conversion tool
PYTHONPATH=${MEGATRON_LM_PATH} python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --no-gradient-accumulation-fusion \
    --hf-checkpoint /root/Qwen3-4B \
    --save /root/Qwen3-4B_torch_dist
```

<!-- cell:17 -->

<a id="3-launch-grpo-training"></a>

### 3.启动GRPO培训

此步骤准备并启动 GRPO 训练运行时。

由于 slime 训练脚本设计为在独立环境中运行，因此它们可能包含在 Jupyter Notebook中运行时不安全的防护措施（例如 `pkill -9 python`）。此外，某些卸载行为可能会导致交互式环境中 AMD GPU 不稳定。

为了确保稳定的基于 Jupyter 的工作流程，下一个单元将执行以下操作：

1. 防止训练脚本终止 Jupyter 内核。
2. 插入训练和部署所需的 `--no-offload` 标志。
3. 减少部署次数，使早期性能改进更容易观察到。

**注意**：您可以根据数据集大小和训练目标调整 `--num-rollout`。 `--num-rollout` 的值越大，每次迭代的推出次数就越多，从而有效地增加训练周期并以更长的运行时间为代价提高收敛性。

运行以下命令作为一次性设置任务来修补训练脚本。

<details markdown="1">
<summary>查看英文原文 · 单元 17</summary>

### 3. Launch GRPO training

This step prepares and launches the GRPO training runtime.

Because slime training scripts are designed to run in standalone environments, they might include safeguards (such as `pkill -9 python`) that are unsafe when run inside a Jupyter notebook. Additionally, certain offloading behaviors can cause instability on AMD GPUs in interactive environments.

To ensure a stable Jupyter-based workflow, the next cell performs the following actions:

1. Prevents the training script from terminating the Jupyter kernel.
2. Inserts required `--no-offload` flags for both training and rollout.
3. Reduces the rollout count to make early performance improvements easier to observe.

**Note**: You can adjust `--num-rollout` based on your dataset size and training goals. A larger value for `--num-rollout` results in more rollouts per iteration, effectively increasing the training epoch and improving convergence at the cost of a longer runtime.

Run the following commands as a one-time setup task to patch the training script.

</details>

```python
%%bash
# Navigate to the slime repository
cd /workspace/notebooks/slime

SCRIPT=scripts/run-qwen3-4B-amd.sh

echo "Patching $SCRIPT ..."

# 1. Comment out `pkill -9 python` only if it is not already commented
if grep -qE '^[[:space:]]*pkill -9 python' "$SCRIPT"; then
  echo " - Commenting out pkill -9 python"
  sed -i 's/^[[:space:]]*pkill -9 python/# pkill -9 python/' "$SCRIPT"
else
  echo " - pkill already commented or not present"
fi

# 2. Inject no-offload flags only if they are not already present
if ! grep -q -- '--no-offload-train' "$SCRIPT"; then
  echo " - Injecting --no-offload flags after --colocate"
  sed -i '/--colocate/a \   --no-offload-train \\\n   --no-offload-rollout \\' "$SCRIPT"
else
  echo " - no-offload flags already present"
fi

sed -i 's/--num-rollout[[:space:]]\+[0-9]\+/--num-rollout 200/' "$SCRIPT"

echo "Patch completed."
```

<!-- cell:19 -->

修补脚本后，启动 GRPO 训练循环：

<details markdown="1">
<summary>查看英文原文 · 单元 19</summary>

After the script is patched, start the GRPO training loop:

</details>

```python
%%bash
# Navigate to the slime repository
cd /workspace/notebooks/slime

# Launch the training script with environment variables set
SLIME_DIR=/root \
MODEL_DIR=/root \
DATA_DIR=/root \
bash scripts/run-qwen3-4B-amd.sh
```

<!-- cell:21 -->

<a id="4-understanding-the-training-script"></a>

### 4. 理解训练脚本

`run-qwen3-4B-amd.sh` 脚本包含 GRPO 训练的所有配置。它被组织成几个参数组，控制训练管道的不同方面。

以下是该脚本关键组件的细分。每个部分对应于培训工作流程的一个特定方面。

<details markdown="1">
<summary>查看英文原文 · 单元 21</summary>

### 4. Understanding the training script

The `run-qwen3-4B-amd.sh` script contains all the configuration for GRPO training. It's organized into several parameter groups that control different aspects of the training pipeline.

Below is a breakdown of the script's key components. Each section corresponds to a specific aspect of the training workflow.

</details>

<!-- cell:22 -->

<a id="model-configuration"></a>

#### 型号配置

```bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-4B.sh"
```

这将从 `scripts/models/qwen3-4B.sh` 加载模型架构设置。这些是定义模型结构的 Megatron-LM 参数。

**重要**：确保 `--rotary-base` 等设置与您的 目标模型 匹配。不同的型号可能使用不同的旋转基值。如有必要，请在使用以下命令获取配置后覆盖此设置：

```bash
source "${SCRIPT_DIR}/models/qwen3-4B.sh"
MODEL_ARGS+=( --rotary-base 10000 )
```

<details markdown="1">
<summary>查看英文原文 · 单元 22</summary>

#### Model configuration

```bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-4B.sh"
```

This loads the model architecture settings from `scripts/models/qwen3-4B.sh`. These are Megatron-LM parameters that define the model structure.

**Important**: Ensure settings such as `--rotary-base` match your target model. Different models might use different rotary base values. If necessary, override this setting after sourcing the configuration using the following commands:

```bash
source "${SCRIPT_DIR}/models/qwen3-4B.sh"
MODEL_ARGS+=( --rotary-base 10000 )
```

</details>

<!-- cell:23 -->

<a id="checkpoint-configuration"></a>

#### 检查点 配置

```yaml
CKPT_ARGS=(
   # HF checkpoint required by SGLang; also used for tokenizer
   --hf-checkpoint ${MODEL_DIR}/Qwen3-4B
   # Reference model checkpoint
   --ref-load ${MODEL_DIR}/Qwen3-4B_torch_dist
   # Actor model load directory; if empty, loads from ref_load
   --load ${MODEL_DIR}/Qwen3-4B_slime/
   --save ${MODEL_DIR}/Qwen3-4B_slime/
   --save-interval 20
)
```

这控制训练期间模型加载和保存的位置。

<details markdown="1">
<summary>查看英文原文 · 单元 23</summary>

#### Checkpoint configuration

```yaml
CKPT_ARGS=(
   # HF checkpoint required by SGLang; also used for tokenizer
   --hf-checkpoint ${MODEL_DIR}/Qwen3-4B
   # Reference model checkpoint
   --ref-load ${MODEL_DIR}/Qwen3-4B_torch_dist
   # Actor model load directory; if empty, loads from ref_load
   --load ${MODEL_DIR}/Qwen3-4B_slime/
   --save ${MODEL_DIR}/Qwen3-4B_slime/
   --save-interval 20
)
```

This controls the location where models are loaded from, and saved to, during training.

</details>

<!-- cell:24 -->

<a id="rollout-configuration"></a>

#### 推出配置

```yaml
ROLLOUT_ARGS=(
   # Dataset configuration
   --prompt-data ${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   # Reward model
   --rm-type deepscaler
   
   # Rollout parameters
   --num-rollout 200
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 0.8
   
   # Training batch configuration
   --global-batch-size 256
   --balance-data
)
```

**关键参数**：
- `--num-rollout`：训练的推出总数
- `--n-samples-per-prompt`：根据提示采样的响应（用于 GRPO 中的组相对优势）
- `--rm-type`：奖励模型类型（史莱姆支持多种类型和使用`--custom-rm-path`的自定义模型）

<details markdown="1">
<summary>查看英文原文 · 单元 24</summary>

#### Rollout configuration

```yaml
ROLLOUT_ARGS=(
   # Dataset configuration
   --prompt-data ${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   # Reward model
   --rm-type deepscaler
   
   # Rollout parameters
   --num-rollout 200
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 0.8
   
   # Training batch configuration
   --global-batch-size 256
   --balance-data
)
```

**Key parameters**:
- `--num-rollout`: Total number of rollouts for training
- `--n-samples-per-prompt`: Responses sampled per prompt (used for group-relative advantages in GRPO)
- `--rm-type`: Reward model type (slime supports multiple types and custom models using `--custom-rm-path`)

</details>

<!-- cell:25 -->

<a id="evaluation-configuration"></a>

#### 评估配置

评估任务继承推出设置，但允许您覆盖特定参数：

```yaml
EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime ${DATA_DIR}/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 0.7
)
```

<a id="performance-and-parallelism"></a>

#### 性能和并行性

```yaml
PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)
```

**关键优化**：
- `--use-dynamic-batch-size`：将不同长度的样本打包成微批次，直至达到令牌限制
- `--max-tokens-per-gpu`：每个 GPU 的令牌硬限制

**注意**：即使使用动态打包，史莱姆也能保证严格的每个令牌损失计算。

####GRPO算法参数

```yaml
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)
```

<a id="grpo-algorithm-parameters"></a>

#### 优化器配置

```yaml
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)
```

<a id="optimizer-configuration"></a>

#### SGLang 配置

```yaml
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2  # SGLang tensor parallelism
   --sglang-mem-fraction-static 0.7
)
```

以 `--sglang-` 为前缀的参数直接转发到 SGLang 引擎。

<details markdown="1">
<summary>查看英文原文 · 单元 25</summary>

#### Evaluation configuration

The evaluation task inherits rollout settings but allows you to override specific parameters:

```yaml
EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime ${DATA_DIR}/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 0.7
)
```

#### Performance and parallelism

```yaml
PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)
```

**Key optimizations**:
- `--use-dynamic-batch-size`: Packs samples of varying lengths into micro batches up to the token limit
- `--max-tokens-per-gpu`: Hard limit of tokens per GPU

**Note**: slime guarantees strict per-token loss calculation even with dynamic packing.

#### GRPO algorithm parameters

```yaml
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)
```

#### Optimizer configuration

```yaml
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)
```

#### SGLang configuration

```yaml
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2  # SGLang tensor parallelism
   --sglang-mem-fraction-static 0.7
)
```

Arguments prefixed with `--sglang-` are forwarded directly to the SGLang engine.

</details>

<!-- cell:26 -->

<a id="5-convert-from-megatron-format-to-hugging-face-format-for-post-training-inference"></a>

### 5.从Megatron格式转换为Hugging Face格式以进行后训练推理

使用 slime 进行 RL 训练后，模型检查点以 Megatron-LM 分布式格式保存，该格式不能直接用于标准 推理 框架。要使用 Hugging Face Transformer 或 SGLang 运行 推理，您必须将这些检查点转换回 Hugging Face (HF) 格式。

要将经过训练的威震天 检查点（来自特定训练迭代）转换回 Hugging Face 格式，请使用以下命令：

<details markdown="1">
<summary>查看英文原文 · 单元 26</summary>

### 5. Convert from Megatron format to Hugging Face format for post-training inference

After RL training with slime, the model checkpoints are saved in the Megatron-LM distributed format, which is not directly usable for standard inference frameworks. To run inference with Hugging Face Transformers or SGLang, you must convert these checkpoints back to Hugging Face (HF) format.

To convert the trained Megatron checkpoint (from a specific training iteration) back into Hugging Face format, use these commands:

</details>

```python
%%bash
# Navigate to the slime repository
cd /workspace/notebooks/slime

# Load model configuration arguments
source scripts/models/qwen3-4B.sh

# Locate megatron-core installation path
MEGATRON_LM_PATH=$(pip list | grep megatron-core | awk '{print $NF}')

PYTHONPATH=${MEGATRON_LM_PATH} python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --no-gradient-accumulation-fusion \
    --hf-checkpoint /root/Qwen3-4B \
    --save /root/Qwen3-4B_torch_dist

PYTHONPATH=${MEGATRON_LM_PATH} python tools/convert_torch_dist_to_hf.py \
  --input-dir /root/Qwen3-4B_slime/iter_0000199 \
  --output-dir /root/Qwen3-4B_slime_hf-iter_0000199 \
  --origin-hf-dir /root/Qwen3-4B
```

<!-- cell:28 -->

转换后，`/root/Qwen3-4B_slime_hf_iter_0000199` 中的数据为标准 Hugging Face 格式，可用于[变形金刚 推理](https://huggingface.co/Qwen/Qwen3-4B)、[SGLang 服务](https://docs.sglang.io/basic_usage/send_request.html)，以及进一步评估或 微调。

<details markdown="1">
<summary>查看英文原文 · 单元 28</summary>

After conversion, the data in `/root/Qwen3-4B_slime_hf_iter_0000199` is in the standard Hugging Face format, ready for [transformers inference](https://huggingface.co/Qwen/Qwen3-4B), [SGLang serving](https://docs.sglang.io/basic_usage/send_request.html), and further evaluation or fine-tuning.

</details>

<!-- cell:29 -->

<a id="summary"></a>

## 总结

恭喜！通过使用 slime 完成这个 GRPO 训练教程，您学习了如何在 AMD GPU 上使用 强化学习 训练大型语言模型。

以下是要点：

- **环境设置**：启用 ROCm 且带有 slime 的 Docker 容器提供了完整的 RL 训练环境。
- **检查点 管理**：Hugging Face 和 Megatron 格式之间的转换可实现跨框架的无缝集成。
- **GRPO 训练**：群体相对优势提供稳定的 RL 训练，无需单独的价值网络。
- **可扩展架构**：SGLang 部署生成和 Megatron-LM 策略优化高效协同工作。

<a id="next-steps"></a>

## 后续步骤

1. **使用不同的数据集进行实验**：将 GRPO 应用于其他推理或指令跟踪数据集。
2. **调整超参数**：针对您的特定用例调整 学习率、KL 系数或采样策略。
3. **扩展到更大的模型**：对更大的 Qwen 模型或其他 LLM 架构使用相同的工作流程。
4. **评估经过训练的模型**：在下游任务上测试微调后的模型以衡量改进情况。

<a id="additional-resources"></a>

## 其他资源

- [slime GitHub 存储库](https://github.com/THUDM/slime)
- [SGLang 文档](https://docs.sglang.io/)
- [Megatron-LM 文档](https://github.com/NVIDIA/Megatron-LM)
- [ROCm 文档](https://rocm.docs.amd.com/en/latest/index.html)
- [AMD AI 开发者中心](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/)

<details markdown="1">
<summary>查看英文原文 · 单元 29</summary>

## Summary

Congratulations! By working through this GRPO training tutorial with slime, you learned how to train large language models using reinforcement learning on AMD GPUs.

Here are the key takeaways:

- **Environment setup**: ROCm-enabled Docker containers with slime provide a complete RL training environment.
- **Checkpoint management**: Converting between Hugging Face and Megatron formats enables seamless integration across frameworks.
- **GRPO training**: Group-relative advantages provide stable RL training without requiring a separate value network.
- **Scalable architecture**: SGLang rollout generation and Megatron-LM policy optimization work together efficiently.

## Next steps

1. **Experiment with different datasets**: Apply GRPO to other reasoning or instruction-following datasets.
2. **Tune hyperparameters**: Adjust the learning rate, KL coefficients, or sampling strategies for your specific use case.
3. **Scale to larger models**: Use the same workflow with larger Qwen models or other LLM architectures.
4. **Evaluate trained models**: Test your fine-tuned models on downstream tasks to measure improvement.

## Additional resources

- [slime GitHub repository](https://github.com/THUDM/slime)
- [SGLang documentation](https://docs.sglang.io/)
- [Megatron-LM documentation](https://github.com/NVIDIA/Megatron-LM)
- [ROCm documentation](https://rocm.docs.amd.com/en/latest/index.html)
- [AMD AI Developer Hub](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/)

</details>

```python

```
