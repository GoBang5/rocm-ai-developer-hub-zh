> 原文作者：Wei Cai  
> [AMD 官方教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/fine_tune/slime_qwen3_4B_GRPO.html) · [固定版本源码](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/fine_tune/slime_qwen3_4B_GRPO.ipynb)  
> 上游版本：`0071bfd4bd3bd9525405c09a6ae69c983b09312a` · 同步日期：2026-10-12  
> 翻译状态：机器初译，待逐篇人工校对；代码与命令保持原文，未在 AMD GPU 上运行验证。

[中文 Notebook](../../notebooks/fine_tune/slime_qwen3_4B_GRPO.ipynb) · [英文原文](../../original/fine_tune/slime_qwen3_4B_GRPO.md) · [中英对照](../../bilingual/fine_tune/slime_qwen3_4B_GRPO.md)

---

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

```python
!git clone https://github.com/THUDM/slime.git
%cd slime
# Note -- You can run the latest upstream version. If you want a stable version, check out the following commit ID
!git checkout 0934a0e
# Install the package
!pip install -e .
```

在继续之前，请确认 slime 已正确安装并且对 Python 可见：

```python
# Verify the installation and version of the required libraries
!pip list | grep slime
```

如果输出中出现 `slime`，则您的环境已准备好进行下一步。

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

```python
!hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B
```

接下来，下载用于数学推理任务的训练数据集（`dapo-math-17k`）：

```python
!hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k
```

然后下载 AIME 2024 评估数据集进行基准测试：

```python
!hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /root/aime-2024
```

<a id="2-convert-the-checkpoint-format"></a>

### 2.转换检查点格式

在开始 GRPO 训练之前，请将预训练的 Hugging Face 检查点 转换为 Megatron Core 分布式格式。

需要进行此转换，因为 slime 使用 Megatron-LM 进行训练。 Megatron-LM 期望根据目标并行化策略（即 张量并行 和 流水线并行）布置模型权重。相比之下，Hugging Face 检查点以与框架无关的单进程格式存储权重。

这是针对给定模型和并行配置的一次性预处理步骤。你不需要每次训练都重复这个过程，只要
因为并行度设置保持不变。

运行以下命令来执行转换：

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

修补脚本后，启动 GRPO 训练循环：

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

<a id="4-understanding-the-training-script"></a>

### 4. 理解训练脚本

`run-qwen3-4B-amd.sh` 脚本包含 GRPO 训练的所有配置。它被组织成几个参数组，控制训练管道的不同方面。

以下是该脚本关键组件的细分。每个部分对应于培训工作流程的一个特定方面。

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

<a id="5-convert-from-megatron-format-to-hugging-face-format-for-post-training-inference"></a>

### 5.从Megatron格式转换为Hugging Face格式以进行后训练推理

使用 slime 进行 RL 训练后，模型检查点以 Megatron-LM 分布式格式保存，该格式不能直接用于标准 推理 框架。要使用 Hugging Face Transformer 或 SGLang 运行 推理，您必须将这些检查点转换回 Hugging Face (HF) 格式。

要将经过训练的威震天 检查点（来自特定训练迭代）转换回 Hugging Face 格式，请使用以下命令：

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

转换后，`/root/Qwen3-4B_slime_hf_iter_0000199` 中的数据为标准 Hugging Face 格式，可用于[变形金刚 推理](https://huggingface.co/Qwen/Qwen3-4B)、[SGLang 服务](https://docs.sglang.io/basic_usage/send_request.html)，以及进一步评估或 微调。

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

```python

```
