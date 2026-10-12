> 原文作者：Daniel Huang  
> [AMD 官方教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/aiter_mla_decode_kernel.html) · [固定版本源码](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/gpu_dev_optimize/aiter_mla_decode_kernel.ipynb)  
> 上游版本：`0071bfd4bd3bd9525405c09a6ae69c983b09312a` · 同步日期：2026-10-12  
> 翻译状态：机器初译，待逐篇人工校对；代码与命令保持原文，未在 AMD GPU 上运行验证。

[中文 Notebook](../../notebooks/gpu_dev_optimize/aiter_mla_decode_kernel.ipynb) · [英文原文](../../original/gpu_dev_optimize/aiter_mla_decode_kernel.md) · [中英对照](../../bilingual/gpu_dev_optimize/aiter_mla_decode_kernel.md)

---

<a id="mla-decoding-kernel-of-the-aiter-library-to-accelerate-llm-inference"></a>

# AITER 库的 MLA 解码内核
**作者**：Daniel Huang
**知识水平**：中级

想象一下，您正在 AMD Instinct™ GPU 上部署 大语言模型（例如 DeepSeek-V3/R1），突然解码阶段的多重潜在注意力 (MLA) 成为性能瓶颈。代币生成感觉缓慢，延迟 不断积累，降低了用户体验。这就是 AMD AITER 库发挥作用的地方，它可以显着加速 MLA 解码注意力内核，为您的模型注入新的活力。

AITER 是 AMD 的高性能算子库，针对 AMD Instinct GPU 上的 AI 工作负载进行了优化。在以下情况下它是必不可少的：

- 操作员的表现远远低于理论潜力。

- 特定运营商成为推理瓶颈。

- 您需要针对 AMD Instinct GPU 进行特定于架构的优化。

本教程将指导您逐步集成 AITER MLA 解码注意力内核，以通过 AMD Instinct GPU 增强 LLM 推理。与本机 PyTorch 实现相比，这将在不同的上下文长度下极大地提高内核性能。您将首先设置 MLA 解码注意内核。

**提示**：AITER 库中的内核已集成到流行的 LLM 推理 框架中，例如 vLLM 和 SGLang。这意味着您还可以通过这些框架从 AMD Instinct GPU 上的 AITER 库获得显着的性能提升！

<a id="prerequisites-setting-up-the-acceleration-environment"></a>

## 前置条件：设置加速环境

本教程是使用以下设置进行开发和测试的，建议使用 AMD Instinct GPU 重现相同的模型加速。

<a id="operating-system"></a>

### 操作系统

* **Ubuntu 22.04**：确保您的系统运行的是 Ubuntu 版本 22.04。

<a id="hardware"></a>

### 硬件

* **AMD Instinct GPU**：确保您使用的是支持 ROCm™ 软件的 AMD Instinct GPU，并且您的系统满足[官方要求](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)。

<a id="software"></a>

### 软件

* **ROCm 6.3.1**：按照 [ROCm 安装指南](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html) 安装并验证 ROCm。

  安装后，使用 `amd-smi` 命令确认您的设置。

  ``` bash
  amd-smi
  ```

  此命令列出可用的 AMD GPU 及其状态。
  
  **注意**：对于 ROCm 6.4 及更早版本，请改用 `rocm-smi` 命令。

* **Docker**：对于容器化部署，请确保 Docker 已正确安装和配置。请遵循适用于您的操作系统的 Docker 安装指南。

  **注意**：确保正确配置 Docker 权限。要配置权限以允许非 root 访问，请运行以下命令：

  ``` bash
  sudo usermod -aG docker $USER
  newgrp docker
  ```

  验证 Docker 是否正常工作：

  ``` bash
  docker run hello-world
  ```

<a id="quick-start-development-environment-set-up"></a>

## 快速启动开发环境搭建

本教程使用预构建的 ROCm PyTorch 映像。

<a id="step-1-launch-the-rocm-pytorch-docker-container"></a>

### 第 1 步：启动 ROCm PyTorch Docker 容器

启动 Docker 容器。该映像是一个交钥匙解决方案，具有预先配置的依赖项：

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
  -w /workspace \
  rocm/pytorch:latest
```

**注意**：该命令将当前目录挂载到容器中的`/workspace`目录下，方便文件访问。它允许您在此 Docker 容器中执行所有工作，包括手动安装 AITER，并开始使用以下动手实践示例。

<a id="step-2-launch-jupyter-notebooks-in-the-container"></a>

### 步骤 2：在容器中启动 Jupyter Notebooks
在 Docker 容器内，使用以下命令安装 JupyterLab：

``` bash
pip install jupyter
```

启动 Jupyter 服务器：

``` bash
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**注意**：如果端口`8888`被占用，请指定不同的端口，例如`--port=8890`。将本教程上传到服务器后，本教程的其余部分可以在 Jupyter Notebook 中作为交互式块运行。

<a id="step-3-manually-install-the-aiter-library"></a>

### 步骤3：手动安装AITER库

AITER 是一个快速扩展的库，具有许多强大的功能。要安装 AITER，请使用以下命令：

```python
%%bash
git clone --recursive https://github.com/ROCm/aiter.git
cd aiter
python3 setup.py develop
export PYTHONPATH=$PYTHONPATH:/workspace/aiter
```

**注意**：如果您在环境中运行 Jupyter 和 AITER，请相应地设置 `PYTHONPATH`。

<a id="understanding-the-mla-decode-attention-kernel"></a>

## 理解 MLA 解码注意力内核

您可以在[AITER源代码](https://github.com/ROCm/aiter/blob/main/aiter/mla.py#L114C1-L126C3)中找到MLA解码注意力内核定义。它至少需要八个输入参数，并且可以接受三个附加可选输入。这是 `mla_decode_fwd` 的函数定义，包括参数：

```
def mla_decode_fwd(  
    q,                   # [batch_size, num_heads, kv_lora_rank + qk_rope_dim]  
    kv_buffer,           # [num_pages, page_size, num_heads_kv, qk_head_dim]  
    o,                   # Output buffer [batch_size, num_heads, kv_lora_rank]  
    qo_indptr,           # Query sequence pointer [batch_size + 1]  
    kv_indptr,           # KV sequence pointer [batch_size + 1]  
    kv_indices,          # KV indices [kv_indptr[-1]]  
    kv_last_page_lens,   # Last page sizes [batch_size]  
    max_seqlen_q,        # Maximum query sequence length  
    sm_scale=None,       # Scaling factor (default: 1.0/sqrt(qk_head_dim))  
    logit_cap=0.0,       # (Under development)  
    num_kv_splits=None,  # KV splits (auto-determined)  
): 
```

每个参数都有特定的形状要求，因此正确的配置是获得最佳性能的关键：

* **q** （`torch.tensor` 类型）：这是具有 `[batch_size, num_heads, kv_lora_rank + qk_rope_dim]` 等形状要求的查询张量。
* **kv buffer**（`torch.tensor`类型）：这是总的KV 缓存张量，其形状要求如`[num_pages, page_size, num_heads_kv, qk_head_dim]`，其中`num_heads_kv`在解码阶段始终为`1`，`num_pages`和`page_size`共同表示可分页KV 缓存。当`page_size = 1`时，KV 缓存被设置为原始表示，这浪费了大量的GPU内存。
* **o** （`torch.tensor` 类型）：这是输出缓冲区。 `mla_decode_fwd` 函数会将结果放入 `o`，其形状要求类似于 `[batch_size, num_heads, kv_lora_rank]`。
* **qo_indptr**（`torch.tensor`类型）：这是一个指向每个查询和输出序列起始地址的指针，形状要求如`[batch_size + 1]`。当批次中每个序列的序列长度不同时，使用`qo_indptr`记录该信息，有助于正确处理每个序列。
* **kv_indptr**（`torch.tensor`类型）：这是一个指向每个context/kv序列起始地址的指针，形状要求如`[batch_size + 1]`。一个批次内每个查询序列不同，答案的序列也不同，因此context/kv序列长度也不同。 `kv_indptr` 变量记录此信息以帮助正确处理查询序列的每个上下文/kv。
* **kv_indices** （`torch.tensor` 类型）：包含每个序列的具体 kv 起始索引。它的形状要求如`[kv_indptr[-1]]`。
* **kv_last_page_lens**（`torch.tensor`类型）：这是每个序列的最后一个页面大小，形状要求如`[batch_size]`。
* **max_seqlen_q:** （`torch.tensor` 类型）：这是该批次中所有查询的最大序列长度。
* **sm_scale**（`scalar`类型）：这等于`1.0 / (qk_head_dim**0.5)`，代表尺度点积注意力公式中的分母。
* **logit_cap**：这是一项正在进行的工作，可以忽略。更多信息请参见下面的[注释](https://github.com/ROCm/aiter/blob/main/aiter/mla.py#L128)。
* **num_kv_splits**（`scalar`类型）：该参数可以忽略。它表示分配多少个 GPU 工作组或块来处理 kv，但代码将使用启发式算法确定该值。

<a id="running-a-practical-example"></a>

## 运行一个实际的例子

现在是开始逐步演练的时候了，该演练将使 MLA 解码注意力在您的 Instinct MI300X 上以闪电般的速度运行。

<a id="setting-the-environment"></a>

### 设置环境

首先准备AMD MI300X GPU，CPU备用：

```python
import os
import sys

# Change working directory to the repo
os.chdir("./aiter")  # relative path from the notebook location

# Add current directory (aiter repo root) to Python path
sys.path.insert(0, os.getcwd())

import torch
from aiter.mla import mla_decode_fwd 

# Let's get our hardware ready for the show!
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"All systems go! Running on: {device}")
```

<a id="prepare-the-tensors"></a>

### 准备张量

现在为这次运行准备张量。您将配置以下内容：

- 一批128个序列，使用`batch_size = 128`

- 4096个代币KV 缓存（我们模型的内存），使用`kv_cache_seqlen = 4096`

- 单查询解码，使用`q_seqlen = 1`

```python
# Your performance parameters
batch_size = 128       # How many sequences we're processing
kv_cache_seqlen = 4096 # How far back our model can remember
q_seqlen = 1           # Decoding one token at a time

# Initialize our pointer arrays
qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int, device=device)
kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int, device=device)

# Fill with sequence lengths (simple case: all equal)
seq_lens_qo = torch.full((batch_size,), q_seqlen, dtype=torch.int, device=device)
seq_lens_kv = torch.full((batch_size,), kv_cache_seqlen, dtype=torch.int, device=device)
```

上面的示例代码首先为 `qo_indptr` 和 `kv_indptr` 声明两个缓冲区，然后用 `q_seqlen = 1` 和 `kv_cache_seqlen = 4096` 填充 `seq_lens_qo` 和 `seq_lens_kv`。为简单起见，假设每个序列具有相同的 `q_seqlen ` 和 `kv cache seqlen`。

然后，它通过向 `cumsum` 函数传递 qkv 的序列长度来填充 `kv_indptr` 和 `qo_indptr`，然后通过从前一个值中减去后一个值来计算每个序列的实际长度。这就是高效注意力的“秘密武器”。

```python
# Calculate cumulative lengths - this tells us where each sequence starts
kv_indptr[1:] = torch.cumsum(seq_lens_kv, dim=0)  # KV memory layout
qo_indptr[1:] = torch.cumsum(seq_lens_qo, dim=0) # Query/output layout

# For example: kv_indptr = [0,5,11,18] means:
# Sequence 0: positions 0-4 (length 5)
# Sequence 1: positions 5-10 (length 6)
# Sequence 2: positions 11-17 (length 7)
```

现在准备您的 键值缓存。将此视为模型的工作记忆。

-初始化每个序列的具体kv起始索引和每个序列的kv最后页透镜（大小）。
- 为简单起见，定义 `page_size = 1`，因此每个序列的 kv 最后一页镜头为 `1`。
- 对于本例，将 `kv_indices` 的最大值设置为 `2097152`。这是从 `batch_size * 16384` 计算出来的，等于 `128 * 16384`。这意味着对于 `128` 的 `batch_size`，您可以为每个序列生成最多 `16384` 令牌。

```python
kv_indices = torch.randint(0, 2097152, (kv_indptr[-1].item(),), dtype=torch.int, device=device)
kv_last_page_lens = torch.ones(batch_size, dtype=torch.int, device=device)
```

现在是时候介绍主要输入，即查询张量和 KV 缓存 以及输出缓冲区。它们是 `q`、`kv buffer` 和 `o`：

```python
num_heads = 128        # Number of attention heads
q_head_dim = 128       # Dimension per head
kv_lora_rank = 512     # LoRA rank for KV
qk_rope_head_dim = 64  # Rotary embedding dimension

# The query tensor - what we're asking our model
q = torch.randn(
    (batch_size * q_seqlen, num_heads, kv_lora_rank + qk_rope_head_dim),
    dtype=torch.bfloat16, device=device
)
num_heads_kv = 1
page_size = 1
q_head_dim = 128
# Our KV cache - the model's knowledge bank
kv_buffer = torch.randn(
    (2097152, page_size, num_heads_kv, kv_lora_rank + qk_rope_head_dim), 
    dtype=torch.bfloat16, device=device
)

# The output buffer - where the magic will happen
o = torch.empty(
    (batch_size * q_seqlen, num_heads, kv_lora_rank), 
    dtype=torch.bfloat16, device=device
).fill_(-1)
```

**注意**：您不必定义这些缓冲区。但是，请确保您定义的形状大小与此处看到的值相匹配。

<a id="launching-the-kernel"></a>

### 启动内核

一切就绪后，启动优化的 MLA 解码注意力内核。

```python
mla_decode_fwd(
        q,
        kv_buffer,
        o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        1,
        sm_scale= 1.0 / (q_head_dim**0.5)
    )
```

现在看看你得到了什么结果。

```python
print(o)
```

最终的形状是：

```python
print(o.shape)
```

<a id="summary"></a>

## 总结

现在，注意力计算已得到优化，结果已准备好无缝流入模型的下一层，从而使整个 推理 管道以最大速度运行。

严格的基准测试显示了内核的真正能力：

**基准亮点**：

- 评估多个上下文长度（512-4096 个标记）
- 使用固定批量大小进行测试 (128)
- 比较不同的MLA算法实现

**结果**：

- 比原生 PyTorch 实现持续加速。

![与aiter的性能比较](../../original/assets/aiter-performance-comparison.png)

想象一下这些收益对您的应用程序意味着什么：
- 减少了实时应用程序的 延迟
- 增加了吞吐量用于批量处理
- 全面降低计算成本

准备好采取下一步了吗？使用以下资源深入了解 AITER 功能：

- 探索 [AITER GitHub 存储库](https://github.com/ROCm/aiter)。 
- 查看其他优化示例。
- 对存储库加注星标以保持新功能的更新。
