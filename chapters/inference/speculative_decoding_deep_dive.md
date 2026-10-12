> 原文作者：Chang Liu  
> [AMD 官方教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/speculative_decoding_deep_dive.html) · [固定版本源码](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/inference/speculative_decoding_deep_dive.ipynb)  
> 上游版本：`0071bfd4bd3bd9525405c09a6ae69c983b09312a` · 同步日期：2026-10-12  
> 翻译状态：机器初译，待逐篇人工校对；代码与命令保持原文，未在 AMD GPU 上运行验证。

[中文 Notebook](../../notebooks/inference/speculative_decoding_deep_dive.ipynb) · [英文原文](../../original/inference/speculative_decoding_deep_dive.md) · [中英对照](../../bilingual/inference/speculative_decoding_deep_dive.md)

---

<a id="speculative-decoding"></a>

# 使用 vLLM 进行投机解码
**作者**：Chang Liu
**知识水平**：中级

本教程演示如何通过在 LLM 服务中启用 投机解码 来实现效率加速。它使用 vLLM（最常用的开源 LLM 框架之一）作为服务引擎。本教程对 AMD Instinct™ MI300X GPU 和 AMD ROCm 软件堆栈进行所有基准测试和分析。有关 投机解码 的基本了解（包括使用指南），请参阅 [vLLM 投机解码 博客](https://www.amd.com/en/developer/resources/technical-articles/vllm-x-amd-highly-efficient-llm-inference-on-amd-instinct-mi300x-gpus.html)。 

当使用 投机解码 启用时，vLLM 的速度可提高 2.3 倍。在本教程中，您将使用 Llama-3.1 70B 作为基本模型，使用 Llama-3.1 1B 作为 草稿模型，比较启用和不启用 投机解码 时它们的服务性能。本教程提供了在 AMD MI300X GPU 上重现性能增益的详细步骤。


<a id="prerequisites"></a>

## 前置条件
本教程是使用以下设置进行开发和测试的。

<a id="operating-system"></a>

### 操作系统

* **Ubuntu 22.04**：确保您的系统运行的是 Ubuntu 版本 22.04。

<a id="hardware"></a>

### 硬件
- **AMD Instinct™ GPU**：本教程已在 AMD Instinct MI300X GPU 上进行了测试。确保您使用的是 AMD Instinct GPU 或支持 ROCm 的兼容硬件，并且您的系统满足[官方要求](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)。


<a id="software"></a>

### 软件
* **ROCm 6.2+**：按照[ROCm 安装指南](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html) 安装并验证 ROCm。安装后，使用以下命令确认您的设置：

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
* 确保 Hugging Face API 令牌具有访问 [Meta Llama 检查点](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) 所需的权限和批准。

<a id="data-preparation"></a>

### 数据准备

本教程使用 Hugging Face 中的示例数据集，该数据集是在设置步骤中准备的。


<a id="preparing-the-serving-environment"></a>

## 准备服务环境

<a id="1-pull-the-docker-image"></a>

### 1. 拉取 Docker 镜像

确保您的系统满足[系统要求](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)。

拉取本教程所需的 Docker 映像：

``` bash
docker pull vllm/vllm-openai-rocm:v0.15.0
```

<a id="2-install-and-launch-jupyter"></a>

### 2.安装并启动 Jupyter

在 Docker 容器内，使用以下命令安装 Jupyter：

``` bash
pip install jupyter
```

启动 Jupyter 服务器：

``` bash
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**注意**：在运行上述命令之前，请确保您的系统上尚未使用端口 `8888`。如果是，您可以通过将 `--port=8888` 替换为其他端口号来指定其他端口，例如 `--port=8890`。


<a id="3-provide-your-hugging-face-token"></a>

### 3. 提供您的拥抱令牌

您需要 Hugging Face API 令牌才能访问 Llama-3.1。在 [Hugging Face Tokens](https://huggingface.co/settings/tokens) 生成您的令牌并请求访问 [Llama-3.1 8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)。令牌通常以“hf_”开头。 

有两种方法可以设置 Hugging Face 令牌：运行 Python 脚本或运行 bash 命令。

通过运行 Python 脚本，您可以在 Jupyter Notebook 中运行以下交互式块来设置令牌：

**注意**：取消选中“添加令牌作为 Git 凭证”选项。

```python
from huggingface_hub import notebook_login, HfApi

# Prompt the user to log in
notebook_login()
```

验证您的令牌是否被正确接受。您应该看到以“令牌已成功验证”开头的输出。

```python
from huggingface_hub import HfApi

try:
    api = HfApi()
    user_info = api.whoami()
    print(f"Token validated successfully! Logged in as: {user_info['name']}")
except Exception as e:
    print(f"Token validation failed. Error: {e}")
```

<a id="downloading-the-model-weights-optional"></a>

## 下载模型权重（可选）

您可以使用以下命令提前下载 `\models` 中的模型权重。本教程使用 AMD 优化的 Llama-3.1 70B 作为基本模型，使用 Llama-3.1 1B 作为投机解码草稿模型。

**注意**：此步骤是可选的。如果您选择跳过它，则在服务进程启动时，将根据您指定的命令自动下载 LLM 服务中使用的模型。

```python
# Download Llama-3.1 70B Instruct FP8 KV model weights in local
!hf download \
  amd/Llama-3.1-70B-Instruct-FP8-KV \
  --local-dir /models/amd--Llama-3.1-70B-Instruct-FP8-KV

# Download Llama-3.1 1B Instruct FP8 KV model weights in local
!hf download \
  amd/Llama-3.2-1B-Instruct-FP8-KV \
  --local-dir /models/amd--Llama-3.2-1B-Instruct-FP8-KV
```

<a id="starting-the-serving"></a>

## 开始服务

您可以使用vLLM的在线服务模式来比较禁用和启用投机解码的性能差异。

使用以下命令在未启用 投机解码 的情况下启动服务器。如果8001端口正在使用，请修改端口：

```python
%%bash
export MODEL_PATH=/path/to/model/weights/cache/directory/
export WORK_PATH=/path/to/workspace/
docker run -d --rm \
  --network=host \
  --ipc=host \
  --privileged \
  --cap-add=CAP_SYS_ADMIN \
  --device=/dev/kfd \
  --device=/dev/dri \
  --device=/dev/mem \
  --group-add render \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --name vllm_spec_dec \
  -v $MODEL_PATH:/models \
  -v $WORK_PATH:/work \
  vllm/vllm-openai-rocm:v0.15.0 \
  bash -c "PYTORCH_TUNABLEOP_ENABLED=0 PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS=100 PYTORCH_TUNABLEOP_MAX_WARMUP_DURATION_MS=10 PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE=1024 PYTORCH_TUNABLEOP_FILENAME=afo_tune_device_%d_full.csv HIP_FORCE_DEV_KERNARG=1 VLLM_USE_ROCM_CUSTOM_PAGED_ATTN=1 VLLM_INSTALL_PUNICA_KERNELS=1 TOKENIZERS_PARALLELISM=false RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1 NCCL_MIN_NCHANNELS=112 VLLM_USE_TRITON_FLASH_ATTN=0 VLLM_FP8_PADDING=1 VLLM_FP8_ACT_PADDING=1 VLLM_FP8_WEIGHT_PADDING=1 VLLM_FP8_REDUCE_CONV=1 \
  vllm serve /models/amd--Llama-3.1-70B-Instruct-FP8-KV \
    --swap-space 16 \
    --disable-log-requests \
    --tensor-parallel-size 1 \
    --distributed-executor-backend mp \
    --dtype float16 \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --enable-chunked-prefill=False \
    --max-num-seqs 300 \
    --port 8001"
```

**注意**：此命令将 `$MODEL_PATH` 挂载到容器中的 `/models` 目录，将 `$WORK_PATH` 挂载到容器中的 `/work` 目录。确保在运行 Docker 命令之前将Notebook文件复制到 `$WORK_PATH` 目录，或在启动后上传到 Jupyter Notebook 环境。保存终端输出中提供的令牌或 URL，以便从 Web 浏览器访问Notebook。您可以从 [AI Developer Hub GitHub 存储库](https://github.com/ROCm/gpuaidev) 下载此Notebook。

该脚本需要为 `vllm serve` 命令提供足够的时间来启动 LLM 服务。较大的 LLM 需要更多的加载时间，因此您可能需要根据您的环境调整睡眠时间。另一种方法是在运行此 Jupyter Notebook 的后续步骤之前启动两个 vLLM 服务器容器，然后使用 `curl` 测试确保 `vllm serve` 命令完成。

```python
!sleep 360
```

现在测试容器的两个 `vllm serve` 操作是否准备就绪。如果没有，您应该在 bash 中使用 `sleep` 命令来添加更多时间，直到收到 `curl` 测试的正确响应。

```python
!curl http://localhost:8001/v1/models
```

<a id="setting-the-client"></a>

## 设置客户端

启动服务器后，启动客户端发送请求。对于客户端，使用 SGLang Docker 映像 `lmsysorg/sglang:v0.5.8-rocm700-mi30x`。在您的环境中启动另一个终端，初始化客户端 Docker 容器，并使用以下命令对性能进行基准测试：

```python
%%bash
docker run -it --rm \
  --ipc=host \
  --network=host \
  --privileged \
  --cap-add=CAP_SYS_ADMIN \
  --device=/dev/kfd \
  --device=/dev/dri \
  --device=/dev/mem \
  --group-add render \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size=192g \
  --name spec_dec_client \
  -v $MODEL_PATH:/models \
  -v $WORK_PATH:/work \
  lmsysorg/sglang:v0.5.8-rocm700-mi30x \
  bash -c "python3 -m sglang.bench_serving \
    --backend vllm \
    --dataset-name random \
    --num-prompt 500 \
    --request-rate 1.0 \
    --random-input 8192 \
    --port 8001 \
    --random-output 256 > /work/vllm_base_log"
```

**注意**：就像您对启动服务器的命令所做的那样，修改 `$MODEL_PATH` 和 `$WORK_PATH` 以对应于您的实际模型目录和工作区。

客户端命令运行后，会生成名为 `vllm_base_log`.log 的日志文件。

<a id="enabling-speculative-decoding-in-serving"></a>

## 在服务中启用 投机解码

在运行下面的单元之前，请停止运行未启用 投机解码 的服务器的上一个单元。
使用以下命令启动启用 投机解码 的服务器。如果8001端口正在使用，请修改端口：

```python
docker run -d --rm \
  --network=host \
  --ipc=host \
  --privileged \
  --cap-add=CAP_SYS_ADMIN \
  --device=/dev/kfd \
  --device=/dev/dri \
  --device=/dev/mem \
  --group-add render \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --name vllm_spec_dec \
  -v $MODEL_PATH:/models \
  -v $WORK_PATH:/work \
  vllm/vllm-openai-rocm:v0.15.0 \
  bash -c "PYTORCH_TUNABLEOP_ENABLED=0 \
  PYTORCH_TUNABLEOP_TUNING=0 \
  PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS=100 \
  PYTORCH_TUNABLEOP_MAX_WARMUP_DURATION_MS=10 \
  PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE=1024 \
  PYTORCH_TUNABLEOP_FILENAME=afo_tune_device_%d_full.csv \
  HIP_FORCE_DEV_KERNARG=1 \
  VLLM_USE_ROCM_CUSTOM_PAGED_ATTN=1 \
  VLLM_INSTALL_PUNICA_KERNELS=1 \
  TOKENIZERS_PARALLELISM=false \
  RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1 \
  NCCL_MIN_NCHANNELS=112 \
  VLLM_USE_TRITON_FLASH_ATTN=0 \
  VLLM_FP8_PADDING=1 \
  VLLM_FP8_ACT_PADDING=1 \
  VLLM_FP8_WEIGHT_PADDING=1 \
  VLLM_FP8_REDUCE_CONV=1 \
  vllm serve /models/amd--Llama-3.1-70B-Instruct-FP8-KV \
    --swap-space 16 \
    --disable-log-requests \
    --tensor-parallel-size 1 \
    --distributed-executor-backend mp \
    --dtype float16 \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --enable-chunked-prefill=False \
    --max-num-seqs 300 \
    --port 8001 \
    --speculative-model /models/amd--Llama-3.2-1B-Instruct-FP8-KV \
    --num_speculative_tokens 5 \
    --speculative-model-quantization fp8"
```

该脚本需要为 `vllm serve` 命令提供足够的时间来启动 LLM 服务。较大的 LLM 需要更多的加载时间，因此您可能需要根据您的环境调整睡眠时间。另一种方法是在运行此 Jupyter Notebook 的后续步骤之前启动两个 vLLM 服务器容器，然后使用 `curl` 测试确保 `vllm serve` 命令完成。

```python
!sleep 360
```

现在测试容器的两个 `vllm serve` 操作是否准备就绪。如果没有，您应该在 bash 中使用 `sleep` 命令来添加更多时间，直到收到 `curl` 测试的正确响应。

```python
!curl http://localhost:8001/v1/models
```

<a id="sending-requests"></a>

## 发送请求

启动服务器后，使用相同的客户端Docker容器`spec_dec_client`发送请求。启动前请确保前一个容器已停止。对于客户端容器，请使用 SGLang Docker 映像 `lmsysorg/sglang:v0.5.8-rocm700-mi30x`。

在终端中，使用以下命令发送请求：

```python
%%bash
docker run -d --rm \
  --ipc=host \
  --network=host \
  --privileged \
  --cap-add=CAP_SYS_ADMIN \
  --device=/dev/kfd \
  --device=/dev/dri \
  --device=/dev/mem \
  --group-add render \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size=192g \
  --name spec_dec_client \
  -v $MODEL_PATH:/models \
  -v $WORK_PATH:/work \
  lmsysorg/sglang:v0.5.8-rocm700-mi30x \
  bash -c "python3 -m sglang.bench_serving \
    --backend vllm \
    --dataset-name random \
    --num-prompt 500 \
    --request-rate 1.0 \
    --random-input 8192 \
    --port 8001 \
    --random-output 256 > /work/vllm_spec_dec_log"
```

为了更深入地探究投机解码，您可以修改上述命令中各个参数的值（例如`request-rate`）来观察性能变化。有关更多信息，请参阅 [投机解码 - 深入探究](https://rocm.blogs.amd.com/software-tools-optimization/speculative-decoding---deep-dive/README.html) 博客中的研究。
