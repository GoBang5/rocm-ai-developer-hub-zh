> 原文作者：Shekhar Pandey and Liz Li  
> [AMD 官方教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/llama4_profiling_vllm.html) · [固定版本源码](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/gpu_dev_optimize/llama4_profiling_vllm.ipynb)  
> 上游版本：`0071bfd4bd3bd9525405c09a6ae69c983b09312a` · 同步日期：2026-10-12  
> 翻译状态：机器初译，待逐篇人工校对；代码与命令保持原文，未在 AMD GPU 上运行验证。

[中文 Notebook](../../notebooks/gpu_dev_optimize/llama4_profiling_vllm.ipynb) · [英文原文](../../original/gpu_dev_optimize/llama4_profiling_vllm.md) · [中英对照](../../bilingual/gpu_dev_optimize/llama4_profiling_vllm.md)

---

<a id="profiling-llama-4-inference-with-vllm"></a>

# 使用 vLLM 分析 Llama-4 推理性能
**作者**：Shekhar Pandey and Liz Li
**知识水平**：中级

分析对于了解 大语言模型 推理 管道中的性能瓶颈至关重要。本教程将引导您完成在具有 ROCm 的 AMD GPU 上使用 vLLM 框架对 **Llama-4 Scout-17B-16E-Instruct** 模型进行分析的过程。您将捕获详细的内核跟踪，然后使用 Perfetto 将其可视化。

<a id="prerequisites"></a>

## 前置条件
在开始本教程之前，请确保您具备以下条件：

- 访问门控 **Llama-4 Scout-17B-16E-Instruct** 模型
- 访问 **Perfetto UI** 

<a id="hardware"></a>

### 硬件

* **AMD GPU**：确保您使用的是 AMD GPU，例如 Instinct™ MI300X 或 Radeon Pro W7900，支持 ROCm，并且您的系统满足[官方要求](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)。

<a id="software"></a>

### 软件

* **ROCm 6.3、6.4**：按照 [ROCm 安装指南](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html) 安装并验证 ROCm。通过运行以下命令验证您的 ROCm 安装：

   ``` bash
   amd-smi
   ```
   
   此命令列出了您的 AMD GPU 以及相关详细信息。
   
   **注意**：对于 ROCm 6.4 及更早版本，请改用 `rocm-smi` 命令。

* **Docker**：确保 Docker 已正确安装和配置。更多信息请参见[Docker安装指南](https://docs.docker.com/get-docker/)。

<a id="prepare-the-training-environment"></a>

## 准备训练环境

请按照以下步骤配置您的教程环境：

<a id="1-pull-the-docker-image"></a>

### 1. 拉取 Docker 镜像

确保您的系统满足[系统要求](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)。

拉取本教程所需的 Docker 映像：

``` bash
docker pull rocm/vllm-dev:main
```

<a id="2-launch-the-docker-container"></a>

### 2.启动Docker容器

在服务器上的终端中启动 Docker 容器并映射必要的目录。

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

<a id="step-by-step-process"></a>

## 分步过程

按照以下步骤分析 Llama-4 模型并捕获内核跟踪。

<a id="step-1-logging-in-to-hugging-face"></a>

### 步骤一：登录Hugging Face
提供您的拥抱令牌

您需要 Hugging Face API 令牌才能访问 Llama-4。在 [Hugging Face Tokens](https://huggingface.co/settings/tokens) 生成您的令牌并请求访问 [Llama-4-Scout 17B-16E-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct)。令牌通常以“hf_”开头。 

在 Jupyter Notebook中运行以下交互式块来设置令牌：

**注意**：取消选中“添加令牌作为 Git 凭证”选项。

```python
from huggingface_hub import notebook_login, HfApi

# Prompt the user to log in
notebook_login()
```

验证您的令牌是否被正确接受：

```python
from huggingface_hub import HfApi

try:
    api = HfApi()
    user_info = api.whoami()
    print(f"Token validated successfully! Logged in as: {user_info['name']}")
except Exception as e:
    print(f"Token validation failed. Error: {e}")
```

<a id="step-2-start-the-vllm-server-with-a-profiler-configuration"></a>

### 步骤 2：使用分析器配置启动 vLLM 服务器
在 JupyterLab 会话中打开一个新的终端选项卡。
在这个新终端中，运行以下命令。保持终端打开。

``` bash
mkdir -p /profile
export VLLM_TORCH_PROFILER_DIR=/profile

<a id="start-the-vllm-server-with-standard-configs"></a>

# Start the vLLM server with standard configs
RCCL_MSCCL_ENABLE=0 \
VLLM_USE_V1=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_USE_MODELSCOPE=False \
VLLM_USE_TRITON_FLASH_ATTN=0 \
vllm serve meta-llama/Llama-4-Scout-17B-16E-Instruct \
    --disable-log-requests \
    -tp 8 \
    --max-num-seqs 64 \
    --no-enable-prefix-caching \
    --max_num_batched_tokens=320000 \
    --max-model-len 32000
```

<a id="step-3-run-the-benchmark-and-capture-the-trace"></a>

### 步骤 3：运行基准测试并捕获跟踪

在服务器运行时，触发综合基准测试请求以生成流量并收集分析数据：

```python
!vllm bench serve \
    --model meta-llama/Llama-4-Scout-17B-16E-Instruct \
    --dataset-name random \
    --random-input-len 2000 \
    --random-output-len 10 \
    --max-concurrency 64 \
    --num-prompts 64 \
    --ignore-eos \
    --percentile_metrics ttft,tpot,itl,e2el \
    --profile
```

确保添加 `--profile` 标志，该标志会在基准测试开始之前启动探查器，然后在基准测试开始后停止并生成跟踪。

**注意：** 上述单元执行完成后，按 **Ctrl-C** 停止您在另一个终端中启动的 vLLM 服务器。

<a id="step-4--visualize-the-trace-using-perfetto-ui"></a>

### 第 4 步：使用 Perfetto UI 可视化跟踪数据

生成跟踪文件后，先将它解压。跟踪数据以 JSON 格式保存。可以使用 [Perfetto UI](https://ui.perfetto.dev/) 查看数据；它是一款面向大规模性能分析数据的强大跟踪查看器，有助于发现推理流水线中的延迟瓶颈、CPU 与 GPU 的执行重叠，以及内核层面的低效操作。按以下步骤打开跟踪文件：

1. 打开 [https://ui.perfetto.dev](https://ui.perfetto.dev/)。
2. 单击 **“Open trace file”**。
3. 上传 `.json` 文件。

打开后，界面大致如下：

![image_0](../../original/assets/image_0.png)

<a id="understanding-the-prefill-and-decode-timelines"></a>

#### 理解预填充与解码时间线

放大跟踪视图，观察各个时间片如何反映 Llama-4 的执行阶段。下面是 Perfetto 中两个重要进程时间线的局部视图。

**注意：**本节主要关注前两条轨道：`python3 3906` 和 `python3 2`。

- `python3 3906`：异步 CPU 调用

  `python3 3906` 下方的**橙色时间片**表示 CPU 上高层模型代码的执行，其中包括：

  - `execute_model`、`forward` 和 `hipMemcpyWithStream` 等调用。
  - `aten::to` 和 `aten::copy_` 等 PyTorch 内部操作。
  - `Llama4ForCausalLM` 的实际前向传播和内存传输。

  该时间片反映了 CPU 对推理过程的组织与调度，从准备输入到将内核派发到 GPU。

- `python3 2`：GPU 内核时间线

  `python3 2` 下方的**粉色时间片**展示 GPU 内核的执行，表示 CPU 将任务加入队列后，GPU 实际完成的计算工作。

  这里有两个关键观察：

  - 两组密集的内核执行之间存在明显间隔。
  - 这个间隔将执行分为两个阶段：
    - **间隔之前**是“预填充”阶段：编码初始提示词，并填充注意力与缓存状态。
    - **间隔之后**是“解码”阶段：模型生成 Token，通常复用已经缓存的键值张量。

<a id="understanding-kernel-timelines"></a>

#### 理解内核时间线

在 Perfetto 中展开 `python3 2` 时间线，会看到两条不同的 GPU 流，分别对应设备上不同类型的操作。

![image](../../original/assets/image_1.png)

**注意：**这个放大视图有助于区分计算内核和通信操作。

- 流 `3 3`：所有 GPU 内核

  流 `3 3` 是**主要计算流**，所有 GPU 内核的执行都在这里调度，包括：

  - `MatMul` 和 GEMM 操作。
  - 融合的 MLP 与注意力模块。
  - 位置编码与 LayerNorm。
  - 各种融合内核或逐元素内核。

  这条流上的内核十分密集，承载了模型的大部分推理活动。内核的执行节奏与间隔有助于诊断：

  - 张量并行各 rank 之间的负载是否均衡。
  - 内核启动之间的空隙。
  - 预填充与解码阶段的区别，可根据间隔前后的内核密度判断。

- 流 `3 8`：AllGather 内核

  流 `3 8` 专门执行 **AllGather** 操作，它们属于张量并行的通信过程。这些内核：

  - 在多 GPU 配置中同步设备之间的激活值。
  - 通常出现在层与层之间。
  - 在 `tp=8` 配置中至关重要，用于同步八个分片的局部输出。

<a id="step-5-zoom-in-to-analyze-the-attention-forward-kernel"></a>

### 第 5 步：放大视图，分析注意力前向内核

注意力是所有 Transformer 语言模型的核心，Llama-4 也不例外。分析它的**注意力前向内核**能够帮助我们深入了解计算效率。下面深入跟踪视图，在内核层面检查执行情况。

<a id="zoom-into-the-kernel-timeline"></a>

#### 放大内核时间线

- 在 Perfetto UI 中先找到 `python3 2`，再找到 `stream 3 3`。
- 用鼠标滚轮或拖动操作，放大一组密集的内核。
- 将鼠标悬停在一个显眼的内核上，应当能看到 `_fwd_kernel` 这样的标签。

下面是跟踪视图中的一个示例：

![image](../../original/assets/image_2.png)

<a id="understanding-the-kernel-slice"></a>

#### 理解内核时间片

在跟踪视图下方的**详细信息**面板中，可以读取以下字段：

| 字段 | 值 |
| --- | --- |
| **名称** | `_fwd_kernel` |
| **类别** | `kernel` |
| **流** | `stream 3 (3)` |
| **进程** | `python3 (2)` |
| **持续时间** | `788µs 256ns` |
| **启动延迟** | 距 CPU 调用约 `~819µs` |

这个内核属于**多头注意力前向传播**，是推理中计算量最大的操作之一。

<a id="tracing-back-to-the-cpu"></a>

#### 回溯到 CPU

在内核详细信息中的前序流附近，可以找到 `hipModuleLaunchKernel`。

单击它可跳回发起该内核启动的 CPU 线程 `python3 3906`，如下图的蓝色部分所示。这个功能特别适合：

- 将 GPU 操作映射回 Python 或 C++ 调用栈。
- 找出派发或同步环节的瓶颈。
- 了解 CPU 将任务加入 GPU 队列所需的时间。

![image](../../original/assets/image_3.png)

按照以上过程，可以追踪各内核在预填充和解码阶段分别耗费的时间。下图给出了某个内核的简要统计。

![image](../../original/assets/image_4.png)

<a id="step-6-programmatic-analysis-and-extracting-the-gpu-kernel-timeline-with-python"></a>

### 第 6 步：使用 Python 进行程序化分析并提取 GPU 内核时间线

除了目视检查，还可以通过程序解析跟踪数据，列出全部 GPU 内核及其持续时间。这种方法有助于追踪：

- 内核启动模式。
- 持续时间的突增。
- 执行中的空隙或异常。

下面是一段简短的 Python 脚本，它读取 `trace.json` 文件，并按开始时间列出所有 GPU 内核：

将 `.gz` 跟踪文件解压缩为 `trace.json`。

```python
!gunzip -c /profile/$(ls /profile | head -n 1) > trace.json
```

运行下面的单元，其中列出了按启动时间排序的所有 GPU 内核。

```python

import json

# Load the trace
with open("trace.json", "r") as f:
    trace = json.load(f)

gpu_kernels = []

# Extract GPU kernel events
for event in trace["traceEvents"]:
    if event.get("ph") != "X":
        continue

    cat = event.get("cat", "").lower()
    name = event.get("name", "")
    start_time = event["ts"]
    duration = event.get("dur", 0)

    if "cuda" in cat or "kernel" in cat:
        gpu_kernels.append({
            "name": name,
            "start": start_time,
            "duration": duration
        })

# Print all GPU kernels with their durations
print(f"{'GPU Kernel':<60} {'Start (us)':<15} {'Duration (us)':<15}")
print("-" * 90)
for k in sorted(gpu_kernels, key=lambda x: x["start"]):
    print(f"{k['name']:<60} {k['start']:<15} {k['duration']:<15}")
```

<a id="conclusion"></a>

## 结论

在本教程中，您演练了使用 vLLM 在 AMD GPU 上分析 Llama-4 推理 的端到端过程，包括：

- 使用 ROCm 和 vLLM 设置容器
- 启用并捕获详细的性能跟踪
- 使用 [Perfetto](https://ui.perfetto.dev/) 可视化 CPU-GPU 交互
- 放大注意力块的内核级活动
- 使用Python以编程方式分析跟踪日志
