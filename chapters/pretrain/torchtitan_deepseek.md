> 原文作者：Shekhar Pandey and Liz Li  
> [AMD 官方教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/torchtitan_deepseek.html) · [固定版本源码](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/pretrain/torchtitan_deepseek.ipynb)  
> 上游版本：`0071bfd4bd3bd9525405c09a6ae69c983b09312a` · 同步日期：2026-10-12  
> 翻译状态：机器初译，待逐篇人工校对；代码与命令保持原文，未在 AMD GPU 上运行验证。

[中文 Notebook](../../notebooks/pretrain/torchtitan_deepseek.ipynb) · [英文原文](../../original/pretrain/torchtitan_deepseek.md) · [中英对照](../../bilingual/pretrain/torchtitan_deepseek.md)

---

<a id="pretraining-with-torchtitan"></a>

# 使用 TorchTitan 预训练模型
**作者**：Shekhar Pandey and Liz Li
**知识水平**：中级

本教程专为有权访问单节点环境（例如 8-GPU 系统）的 [AMD Developer Cloud](https://www.amd.com/en/developer/resources/cloud-access/amd-developer-cloud.html) 用户而设计。
它演示了在 ROCm 上使用 TorchTitan AMD fork 的端到端单盒训练工作流程。

**你将学到什么**

如何：

- 设置干净的单节点训练环境。
- 在 混合专家 (MoE) 配置上启动 TorchTitan 并了解关键标志。
- 监视日志并验证运行。

**TorchTitan**

TorchTitan 是一个紧凑的概念验证，演示了最新的原生 PyTorch 分布式 API。
AMD 分支添加了完整的 ROCm 支持和实验性 Primus-Turbo `FP8` 内核。
本教程使用 [AMD-AGI/torchtitan-amd](https://github.com/AMD-AGI/torchtitan-amd) 存储库的 `dev/primus_turbo` 分支。

**型号**
 
DeepSeek‑MoE 16B Base（大约有 16.4B 参数）是一种 MoE LLM，具有细粒度专家和共享专家隔离。
Hugging Face 模型卡为 [deepseek-moe-16b-base](https://huggingface.co/deepseek-ai/deepseek-moe-16b-base)。

**目标**

通过清晰的日志记录和检查获得成功的单节点运行，以便您稍后可以放心地扩展或交换数据集。

<a id="prerequisites"></a>

## 前置条件

本教程是使用以下设置进行开发和测试的。

<a id="operating-system"></a>

### 操作系统

* **Ubuntu 22.04**：确保您的系统运行的是 Ubuntu 22.04。

<a id="hardware"></a>

### 硬件

* **AMD Instinct™ GPU**：本教程在 AMD Instinct MI300X 节点（具有 8 个 GPU）上进行了验证。确保您使用的是 AMD Instinct GPU 或支持 ROCm 的兼容硬件，并且您的系统满足[官方要求](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)。

<a id="software"></a>

### 软件

* **ROCm 6.3 或更高版本**：按照 [ROCm 安装指南](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html) 安装并验证 ROCm。安装后，使用以下命令确认您的设置：

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

<a id="prepare-the-training-environment"></a>

## 准备训练环境

要设置您的培训环境，请按照下列步骤操作：

<a id="1-pull-the-docker-image"></a>

### 1. 拉取 Docker 镜像

确保您的系统满足 [ROCm 系统要求](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)。

拉取本教程所需的 Docker 映像：

``` bash
docker pull docker.io/rocm/primus:v25.9_gfx942
```

<a id="2-launch-the-docker-container"></a>

### 2.启动Docker容器

启动 Docker 容器并映射必要的目录。

``` bash
docker run -it --rm \
  --name torchtitan_tutorial \
  --device /dev/dri --device /dev/kfd \
  --network host --ipc host --group-add video \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \
  -v "$PWD":/workspace \
  -w /workspace \
  --entrypoint /bin/bash \
  docker.io/rocm/primus:v25.9_gfx942
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
jupyter lab --ip=127.0.0.1 --port=8888 --no-browser --allow-root
```

**注意**：在运行上述命令之前，请确保您的系统上尚未使用端口 `8888`。如果是，您可以通过将 `--port=8888` 替换为其他端口号来指定其他端口，例如 `--port=8890`。

<a id="4-set-up-the-environment"></a>

### 4.设置环境

现在，通过禁用 InfiniBand (IB) 并清除 Slurm 变量，使用单节点默认设置设置环境。

```python
import os

# --- export env vars ---
os.environ["NCCL_SOCKET_IFNAME"] = "lo"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ.pop("SLURM_MASTER_ADDR", None)
os.environ.pop("SLURM_MASTER_PORT", None)

# --- verify ---
print("NCCL_SOCKET_IFNAME =", os.environ["NCCL_SOCKET_IFNAME"])
print("NCCL_IB_DISABLE =", os.environ["NCCL_IB_DISABLE"])
```

<a id="5-clone-and-install-torchtitan-and-primus-turbo-inside-the-container"></a>

### 5. 在容器内克隆并安装 TorchTitan 和 Primus-Turbo

使用 Git 克隆 TorchTitan AMD 分支（或更新它，如果它已经存在）并签出 `Primus-Turbo` 分支。

```python
%%bash
set -euo pipefail

if [ -d "torchtitan-amd/.git" ]; then
  echo "[info] Repo exists: torchtitan-amd"
else
  echo "[info] Cloning https://github.com/AMD-AIG-AIMA/torchtitan-amd.git ..."
  git clone "https://github.com/AMD-AIG-AIMA/torchtitan-amd.git"
fi

cd torchtitan-amd
git checkout dev/primus_turbo
```

为 AITER 和 PyTorch 扩展创建每主机构建缓存，以避免用户之间的权限冲突。

```python
import os
import socket
from pathlib import Path

# --- unset env vars  ---
os.environ.pop("AITER_JIT_DIR", None)
os.environ.pop("TORCH_EXTENSIONS_DIR", None)

# --- hostname  ---
host_name = socket.gethostname()

# --- build dir ---
tmp_build_dir = Path("/workspace/torchtitan-amd/3rdparty/build") / host_name
tmp_build_dir.mkdir(parents=True, exist_ok=True)   # like `mkdir -p ...`

# --- export env vars ---
os.environ["AITER_JIT_DIR"] = str(tmp_build_dir / "single_aiter_cache")
os.environ["TORCH_EXTENSIONS_DIR"] = str(tmp_build_dir / "torch_extensions")

# --- verify ---
print("AITER_JIT_DIR =", os.environ["AITER_JIT_DIR"])
print("TORCH_EXTENSIONS_DIR =", os.environ["TORCH_EXTENSIONS_DIR"])
```

将 TorchTitan 安装在容器内。

```python
%%bash
set -euo pipefail

cd torchtitan-amd

pip install -r requirements.txt
pip install torchao
pip install -e .
```

从车轮上安装 Primus-Turbo。

```python
%%bash
set -euo pipefail

cd torchtitan-amd

pip uninstall numpy -y && pip install numpy==1.26.4
pip3 install --extra-index-url https://test.pypi.org/simple ./3rdparty/primus_turbo-0.1.0+dbeaf79-cp310-cp310-linux_x86_64.whl
```

<a id="6-download-the-deepseekmoe-16b-base-model"></a>

### 6.下载DeepSeekMoE 16B基础模型

从 Hugging Face 下载 `deepseek-ai/deepseek-moe-16b-base` 模型：

```python
%%bash
set -euo pipefail

cd torchtitan-amd
python scripts/download_hf_assets.py --repo_id deepseek-ai/deepseek-moe-16b-base --assets tokenizer
```

<a id="create-the-training-configuration"></a>

## 创建训练配置

按照以下步骤创建配置文件来训练模型。

<a id="1-create-a-working-copy-of-the-toml-configuration-file"></a>

### 1. 创建 TOML 配置文件的工作副本

为 DeepSeekMoE 16B 模型创建基本配置文件的工作副本。

```python
import pathlib, shutil

repo_dir = pathlib.Path("torchtitan-amd")

override_dir = repo_dir / "run_configs"
override_dir.mkdir(parents=True, exist_ok=True)

base_config_path = repo_dir / "torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b.toml"
override_config_path = override_dir / "deepseek_v3_16b_local.toml"

shutil.copyfile(base_config_path, override_config_path)
```

<a id="2-modify-the-file"></a>

### 2.修改文件

安装`tomlkit`，一个修补TOML文件的工具。

```python
!pip install tomlkit
```

接下来，修改配置文件以适合您的训练。
对于本教程，请选择以下设置：

```toml
[profiling]
enable_profiling = false     # Profiling disabled. Enable it if you need the traces.

[training]
local_batch_size = 32        # Edit for a different batch size.
steps = 10                   # Training for only 10 steps. Increase this for real training.
dataset = "c4_test"          # This is a small sample dataset. Use `"c4"` for a real HF dataset.

[parallelism]
expert_parallel_degree = 8   # Edit for a different expert parallel degree.

[checkpoint]
enable = false               # Checkpointing disabled. Enable it for real training.
```

**可选**：针对不同的配置编辑以下块。您也可以修改其他字段。查看下一个单元格的输出以查看配置文件中的所有字段。

```python
import tomlkit

doc = tomlkit.parse(override_config_path.read_text())

# profiling
doc["profiling"]["enable_profiling"]         = False

# training
doc["training"]["local_batch_size"]          = 32
doc["training"]["dataset"]                   = "c4_test"
doc["training"]["steps"]                     = 10

# parallelism
doc["parallelism"]["expert_parallel_degree"] = 8

# checkpoint
doc["checkpoint"]["enable"]                  = False

override_config_path.write_text(tomlkit.dumps(doc))

print("Working config:", str(override_config_path))
```

查看生成的 TOML 文件：

````python
from IPython.display import Markdown, display

text = override_config_path.read_text()
display(Markdown(f"```toml\n{text}\n```"))
````

<a id="3-set-the-environment-variable"></a>

### 3.设置环境变量

设置环境变量 `CONFIG_FILE` 以指向您的工作副本。

```python
import pathlib, os

config_path = override_config_path.resolve().relative_to(pathlib.Path("/workspace/torchtitan-amd"))

# --- export env vars ---
os.environ["CONFIG_FILE"] = str(config_path)

# --- verify ---
print("CONFIG_FILE =", os.environ["CONFIG_FILE"])
```

<a id="pretrain-the-model"></a>

## 预训练模型

本节介绍如何训练模型并验证结果。

<a id="1-start-training"></a>

### 1.开始训练

您现在可以运行训练脚本。
AITER 的构建可能需要几分钟时间。训练开始后，它会打印每个步骤的指标。
要稍后验证运行结果，请将日志保存到 `logs/dsv3_16b_$(date +%Y%m%d_%H%M%S).log`。

```python
%%bash
set -euo pipefail

cd torchtitan-amd
mkdir -p logs

export PYTHONUNBUFFERED=1
stdbuf -oL -eL ./run_train.sh 2>&1 | tee -a logs/dsv3_16b_$(date +%Y%m%d_%H%M%S).log
```

<a id="2-verify-the-run-results"></a>

### 2.验证运行结果

最后，您可以从日志中提取步骤、损失和标记。您应该预期损失会在样本数据的前几步内略有下降，并且标记不为零。

```python
import re, pathlib, pandas as pd, matplotlib.pyplot as plt

log_dir = pathlib.Path("./torchtitan-amd/logs")  # adjust if needed
log_files = sorted(log_dir.glob("**/*.log")) + sorted(log_dir.glob("**/*.out"))
assert log_files, f"No logs found in {log_dir.resolve()}"
lf = max(log_files, key=lambda p: p.stat().st_mtime)  # newest

# Strip ANSI color codes
ansi_re = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

# Tolerant metrics line (handles commas in numbers and both 'tps' or 'tokens/s')
metrics_re = re.compile(
    r"step:\s*(\d+).*?"
    r"loss:\s*([-\d.eE+]+).*?"
    r"grad_norm:\s*([-\d.eE+]+).*?"
    r"memory:\s*([-\d.eE+]+)GiB\(([-\d.eE+]+)%\).*?"
    r"(?:tps|tokens/s):\s*([\d.,eE+-]+).*?"
    r"tflops:\s*([-\d.eE+]+).*?"
    r"mfu:\s*([-\d.eE+]+)%",
    re.IGNORECASE
)

rows = []
for raw in lf.read_text(errors="ignore").splitlines():
    line = ansi_re.sub("", raw)
    m = metrics_re.search(line)
    if m:
        step = int(m.group(1))
        loss = float(m.group(2))
        grad = float(m.group(3))
        mem_gib = float(m.group(4))
        mem_pct = float(m.group(5))
        tps = float(m.group(6).replace(",", ""))  # remove thousands commas
        tflops = float(m.group(7))
        mfu = float(m.group(8))
        rows.append({
            "step": step, "loss": loss, "grad_norm": grad,
            "mem_gib": mem_gib, "mem_%": mem_pct,
            "tps": tps, "tflops": tflops, "mfu_%": mfu
        })

df = pd.DataFrame(rows).sort_values("step").drop_duplicates("step")
display(df.tail(10))

# Quick health checks
if not df.empty:
    print(f"Steps parsed: {df.step.min()}..{df.step.max()} ({len(df)} points)")
    if len(df) >= 2 and df["loss"].iloc[-1] <= df["loss"].iloc[0]:
        print("✓ Loss decreased at least once — smoke test looks healthy.")
    else:
        print("! Loss didn’t drop yet — tiny sample data can wobble; let a few more steps run.")

# Plots
plt.figure()
plt.plot(df["step"], df["loss"])
plt.xlabel("step"); plt.ylabel("loss"); plt.title("Loss vs step")
plt.show()

plt.figure()
plt.plot(df["step"], df["tps"])
plt.xlabel("step"); plt.ylabel("tokens/sec (tps)"); plt.title("Throughput vs step")
plt.show()

# Optional: confirm training ended
finished = "Training completed" in ansi_re.sub("", lf.read_text(errors="ignore"))
print("Finished flag:", "✓" if finished else "—")
```
