> 原文作者：Vicky Tsang, Dipto Deb  
> [AMD 官方教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/t2v_comfyui_api_mode_instinct.html) · [固定版本源码](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/inference/t2v_comfyui_api_mode_instinct.ipynb)  
> 上游版本：`0071bfd4bd3bd9525405c09a6ae69c983b09312a` · 同步日期：2026-10-12  
> 翻译状态：机器初译，待逐篇人工校对；代码与命令保持原文，未在 AMD GPU 上运行验证。

[中文 Notebook](../../notebooks/inference/t2v_comfyui_api_mode_instinct.ipynb) · [英文原文](../../original/inference/t2v_comfyui_api_mode_instinct.md) · [中英对照](../../bilingual/inference/t2v_comfyui_api_mode_instinct.md)

---

<!--
  SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
  SPDX-License-Identifier: Apache-2.0
-->

<a id="running-comfyui-generative-workflows-from-python-on-amd-instinct-gpus"></a>

# 在 AMD Instinct GPU 上通过 Python 运行 ComfyUI 生成工作流
**作者**：Vicky Tsang, Dipto Deb
**知识水平**：中级  

[![在 AMD Developer Cloud 中打开](../../original/assets/jupyter_notebook_button_fnl_small.png)](https://notebooks.amd.com/tutorials/t2v_comfyui_api_mode_instinct.ipynb)

本教程演示如何在运行 [ROCm™](https://rocm.docs.amd.com/) 软件堆栈的 AMD Instinct™ 数据中心 GPU 上完全从 Python（不使用浏览器）驱动 [ComfyUI](https://www.comfy.org/) 生成式 AI 工作流程。您无需使用图形界面，只需通过 HTTP API 将工作流程作为 JSON 文档提交到 ComfyUI 服务器，等待任务完成，然后检索结果。这种无头、可编写脚本的方法是您在使用不带显示器的远程服务器时所需要的，也是在 HPC 和云基础设施上自动执行脚本、CI 或批处理作业中的生成管道的基础。

本教程对相同的输入照片运行两个实用的工作流程：第一个使用 **Hunyuan3D v2.1** 模型将照片转换为 3D 网格，第二个使用 **Wan2.2 5B** 模型将照片转换为短视频。使用 Jupyter Notebook，您可以从 Python 提交每个任务，并直接在Notebook中查看输出。  

本教程假设您之前没有使用 ComfyUI 的经验。如果本教程中的某些术语对您来说很陌生，请不要担心 - 下面的[关键概念](#key-concepts) 部分将以通俗易懂的语言解释这些术语。

> **配套教程：** 此Notebook是 [*使用 ComfyUI 和 AMD Radeon GPU 生成文本到视频*](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/t2v_comfyui_radeon.html) 的后续产品。如果您是 ComfyUI 的新手，那么 Radeon 教程是一个很好的起点：它通过引导您完成 **交互式浏览器 UI** 来熟悉 ComfyUI（请参见下面的比较表）。
>
> 两个教程都在 AMD GPU 上使用 ComfyUI，但它们针对不同的受众和工作流程：
>
> <table>
> <thead><tr style="border-bottom: 2px solid #888;"><th></th><th>Radeon教程</th><th>本教程</th></tr></thead>
> <tbody>
> <tr><td><strong>H硬件</strong></td><td>消费者<strong>AMD Radeon</strong> GPU（例如RX 7900 XTX)</td><td>数据中心 <strong>AMD Instinct</strong> GPU (MI300X / MI325X / MI350X / MI355X)</td></tr>
> <tr><td><strong>如何在 <strong> 浏览器中交互式驾驶 ComfyUI</strong></td><td> UI</strong></td><td><strong>无头，来自 Python</strong> 通过 HTTP API（无浏览器）</td></tr>
> <tr><td><strong>所示工作流程</strong></td><td>A单个文本到视频工作流程</td><td><strong>Two</strong> 工作流程：图像到 3D 和图像到视频</td></tr>
> <tr><td><strong>最适合</strong></td><td>亲自动手学习ComfyUI desk</td><td><strong>在远程服务器和管道中自动化</strong>生成工作流程</td></tr>
> </tbody>
> </table>
>
> 简而言之：Radeon 教程教您如何交互地使用 ComfyUI；这将向您展示如何在服务器级硬件上**将其作为服务进行操作**。
> **配套博客文章：** 有关 AMD Instinct GPU 上的许多 ComfyUI 工作流程的性能信息，请参阅 AMD ROCm 博客 [*使用 ROCm 加速 AMD Instinct MI355X GPU 上的 ComfyUI 工作流程*](https://rocm.blogs.amd.com/artificial-intelligence/comfyui/README.html)。

<a id="what-you-will-learn"></a>

### 你将学到什么

**核心技能**：在 AMD Instinct 上以编程方式驱动 ComfyUI

- 如何启动 ComfyUI 服务器并确认 ROCm + PyTorch 运行时。
- ComfyUI 的 HTTP API 如何工作，以及如何使用一些小助手来提交工作流程、轮询完成情况以及检索输出。
- 如何从 Python 运行两个真实的扩散工作流程并显示其输出：
  - *图像转3D*使用**[Hunyuan3D v2.1](https://huggingface.co/tencent/Hunyuan3D-2.1)**: 将照片转换为3D网格(`.glb`)。
  - *图像到视频*使用**[Wan2.2 5B](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged)**: 将照片变成短MP4相机轨道。

**可选技能**：衡量绩效

- 如何将工作流程包装在小型线束中并在您自己的硬件上测量 吞吐量 **。

<a id="tutorial-at-a-glance"></a>

### 教程一览

本教程由三个部分组成： 

- **第 1 部分：设置 ComfyUI。** 在 ROCm 容器内启动 ComfyUI 服务器并加载 API 帮助程序。  
- **第 2 部分：使用两个扩散工作流程生成。** 在三张照片上运行图像到 3D 和图像到视频工作流程并可视化结果。  
- **第 3 部分：测量 吞吐量。** *（可选）* 在您自己的 GPU 上对工作流程进行基准测试。

<a id="key-concepts"></a>

## 关键概念

本节介绍本教程涉及的主要技术概念。  

<a id="diffusion-models"></a>

### 扩散模型

**扩散模型**是大多数现代图像、视频和 3D 生成背后的生成人工智能模型。他们学习从随机噪声开始，并在提示或输入图像的指导下迭代地将其“去噪”成连贯的结果。每次迭代称为*步骤*。

<a id="comfyui-and-workflows"></a>

### ComfyUI 和工作流程

**[ComfyUI](https://www.comfy.org/)** 是一个开源的、**基于节点的**接口，用于扩散模型。您可以通过将节点（“加载模型”、“编码提示”、“样本”、“解码”、“保存”）连接在一起来构建管道，形成称为“工作流”的图形。ComfyUI 可以将任何工作流导出到 JSON 文档（称为 API 格式）。本教程直接以 API 格式提交工作流，因此无需为它们使用可视化编辑器。

<a id="driving-comfyui-headlessly-the-http-api"></a>

### 无头驱动 ComfyUI（HTTP API）

ComfyUI 作为长期的 **HTTP 服务器** 运行。除了图形浏览器 UI 之外，它还公开了一个 HTTP API：您将工作流程 `POST` 发送到 `/prompt`，轮询 `/history/<id>` 直至完成，然后从 `/view` 下载生成的文件。这里的“无头”意味着仅使用此 API，不附加显示器——这是在远程数据中心节点上运行的标准方式。

<a id="the-two-workflows"></a>

### 两个工作流程

- *图像到 3D* 使用 **[Hunyuan3D v2.1](https://huggingface.co/tencent/Hunyuan3D-2.1)**: 从一张照片中生成保存为 `.glb` 文件的 3D 网格。
- *图像到视频*使用**[Wan2.2 5B](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged)**:) 从一张照片（加上文本提示）它通过生成每一帧来合成短视频。

<a id="rocm-and-amd-instinct-gpus"></a>

### ROCm 和 AMD Instinct GPU

**[ROCm™](https://rocm.docs.amd.com/)** 是 AMD 用于 GPU 计算的开放软件平台。**AMD Instinct™** GPU（MI300X、MI325X、MI350X、MI355X)是 AMD 的 AI 数据中心加速器。一切都在 ROCm **Docker 容器**内运行，因此环境易于重现。

<a id="prerequisites"></a>

## 前置条件  

本教程是使用以下设置进行开发和测试的。  

<a id="hardware"></a>

### 硬件

- AMD Instinct GPU：本教程在 AMD Instinct **MI300X** 和 **MI355X** 上进行了测试。确保您使用的是 AMD Instinct GPU 或支持 ROCm 的兼容硬件，并且您的系统满足[官方要求](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)。您的 GPU 还需要有足够的 VRAM 来加载下面列出的模型检查点。

<a id="software"></a>

### 软件

下表列出了**本教程所测试的软件的确切版本**。其他版本可能有效，但尚未经过验证。

|组件|版本（已测试）|参考|
|---|---|---|
| ROCm | 7.2.0 | [快速入门安装指南](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html) |
|码头工人 | 24.0.7 | [Docker安装指南](https://docs.docker.com/get-docker/) |
| `rocm/comfyui` 图像 | `comfyui-0.18.2.amd0_rocm7.2.0_ubuntu24.04` | ComfyUI-on-ROCm 图像（在环境设置中拉取）|
| PyTorch | 2.10（ROCm 7.2 版本）|船舶在图像内 |

您**不需要**需要手动构建任何内容 - 本教程中使用的容器映像附带 ComfyUI 和支持 ROCm 的 PyTorch。但是，模型检查点（总共约 27 GB）**未**捆绑，但如果您通过设置 `DOWNLOAD_MODELS = True` 选择加入（请参见 **第 1 部分**中的配置单元），则可以自动从 Hugging Face 下载（请参见 **第 2 部分**）。启用该选项之前，请确保您有足够的可用磁盘空间。

> **提示：** 在开始之前，通过在终端中运行 `docker run hello-world` 来确认 Docker 已安装并且可以在没有 `sudo` 的情况下运行。

<a id="environment-setup"></a>

## 环境设置

本部分在 ComfyUI-on-ROCm 容器内设置Notebook环境。  

**注意**：此Notebook可在 [AI Developer Hub GitHub 存储库](https://github.com/ROCm/gpuaidev) 中获取。

此Notebook中的所有工作流程都在容器**内部**运行，其中 `COMFYUI_PATH` 环境变量指向 ComfyUI 源树（通常为 `/workload/ComfyUI`）。

<a id="1-pull-the-image-and-launch-the-container"></a>

### 1. 拉取镜像并启动容器

拉取预构建的 ComfyUI-on-ROCm 映像并启动一个暴露 AMD GPU 的容器。两个端口转发到主机（对于 JupyterLab 为 **8888**，对于 ComfyUI 服务器为 **8188**），并安装包含此Notebook的目录。

```bash
docker pull rocm/comfyui:comfyui-0.18.2.amd0_rocm7.2.0_ubuntu24.04

<a id="run-from-the-directory-where-you-cloned-the-ai-developer-hub-github-repository"></a>

# Run from the directory where you cloned the AI Developer Hub GitHub repository.
docker run -it --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --shm-size=16g \
  -p 8888:8888 \
  -p 8188:8188 \
  -e COMFYUI_PATH=/workload/ComfyUI \
  -v "$PWD:/work" \
  rocm/comfyui:comfyui-0.18.2.amd0_rocm7.2.0_ubuntu24.04
```

这将在容器**内部**打开一个外壳，该Notebook可在 `/work` 下找到。在那里运行剩余的步骤。

<a id="2-open-this-notebook-in-jupyterlab"></a>

### 2. 在 JupyterLab 中打开此Notebook

该映像不会自动启动 JupyterLab。从容器 shell 中，安装 JupyterLab 并将其启动绑定到所有接口，以便 Docker 的端口转发可以到达它：

```bash
pip install jupyterlab
jupyter lab --no-browser --ip=0.0.0.0 --port=8888 --allow-root --notebook-dir=/work
```

启动时，它会打印一个包含一次性令牌的 URL，格式为 `http://127.0.0.1:8888/lab?token=<long-token>`。

**如果 GPU 位于通过 SSH 访问的远程计算机上**，请在本地系统上打开第二个终端并转发两个端口，以便您可以从本地浏览器访问 ComfyUI 服务器及其浏览器 UI：

```bash
ssh -L 8888:localhost:8888 -L 8188:localhost:8188 <user>@<gpu-host>
```

将打印的 `http://127.0.0.1:8888/lab?token=<long-token>` URL 粘贴到本地浏览器中，然后从 `docs/notebooks/inference/t2v_comfyui_api_mode_instinct.ipynb` 打开此Notebook。要在完成后停止服务器，请在容器 shell 中按 `Ctrl+C` 两次。

<a id="3-confirm-the-gpus-are-visible"></a>

### 3. 确认 GPU 可见

ROCm 包括 `amd-smi`，这是一个命令行工具，列出了容器可以看到的 AMD GPU。运行下面的单元格以确认您的 AMD Instinct GPU 的可见性。如果未找到该命令或未出现 GPU，请在继续之前重新检查容器的 `--device` 标志和 ROCm 安装。

```python
# List the AMD GPUs visible inside the container. You should see your Instinct GPU(s).
# For ROCm 6.4 and earlier, use `rocm-smi` instead of `amd-smi`.
!amd-smi list || rocm-smi
```

<a id="part-1-set-up-and-verify-comfyui-on-instinct"></a>

## 第 1 部分：在 Instinct 上设置并验证 ComfyUI

随着容器运行并在其中打开此Notebook，**第 1 部分** 建立运行时和我们将在本教程的其余部分讨论的服务器。有三个简短的步骤：

- 验证 PyTorch 是否是使用 ROCm 支持构建的，并且它可以识别 GPU，并设置Notebook的工作目录。
- 暂存本教程中使用的输入照片和工作流程文件。
- 将 ComfyUI 服务器作为后台进程启动，并加载一小组将在第 2 部分和第 3 部分中使用的 HTTP API 帮助程序。

<a id="install-the-notebooks-python-dependencies"></a>

### 安装Notebook的Python依赖项

ComfyUI-on-ROCm 映像附带 ComfyUI 和支持 ROCm 的 PyTorch。下面的单元格安装了该Notebook用于可视化的一些附加 Python 包：

- `trimesh` + `fast-simplification`：用于加载和抽取 Hunyuan3D `.glb` 网格以用于转盘预览（**第 2 部分**）。
- `imageio` + `imageio-ffmpeg`：用​​于将预览编码为 MP4（它捆绑了自己的 `ffmpeg`，因此不需要系统级工具）。
- `pillow`：用​​于内联结果网格中的图像处理。

这些工具都不使用 GPU，它们仅渲染输出以供显示。

```python
%pip install --quiet trimesh fast-simplification "imageio[ffmpeg]" pillow
```

<a id="verifying-the-runtime-pytorch-rocm-and-gpu-visibility"></a>

### 验证运行时：PyTorch ROCm 和 GPU 可见性

下一个单元格确认 PyTorch 是使用 ROCm 支持构建的，打印检测到的设备名称，并找到 `$COMFYUI_PATH`。在工作流程部分之前运行它。如果无法验证运行时，请先修复安装，然后再继续。

```python
import os
import sys
import json
import shutil
import time
from copy import deepcopy
from pathlib import Path

import torch

assert torch.cuda.is_available(), (
    "No HIP / CUDA device visible to PyTorch. Check container --device flags and ROCm install."
)
print("python:", sys.version.split()[0])
print("torch: ", torch.__version__, "| HIP:", getattr(torch.version, "hip", None) or "(not a ROCm build)")
print("GPU 0: ", torch.cuda.get_device_name(0))

COMFYUI_PATH = Path(os.environ.get("COMFYUI_PATH", "/workload/ComfyUI"))
assert (COMFYUI_PATH / "main.py").is_file(), (
    f"{COMFYUI_PATH}/main.py not found - set COMFYUI_PATH to your ComfyUI checkout."
)

NOTEBOOK_DIR = Path(os.getcwd()).resolve()  # also where utils_comfyui.py lives
WORKFLOWS_DIR = NOTEBOOK_DIR / "workflows"
ASSETS_DIR = NOTEBOOK_DIR / "assets"
OUTPUTS_DIR = NOTEBOOK_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# Set to True to let Part 2 download the model checkpoints (~27 GB total) from
# Hugging Face into ComfyUI/models/. DOWNLOAD_MODELS is False by default so a first read does
# not trigger a large download; flip it once you are ready to run the workflows.
DOWNLOAD_MODELS = False

print("COMFYUI_PATH:", COMFYUI_PATH)
print("Notebook dir:", NOTEBOOK_DIR)
print("DOWNLOAD_MODELS:", DOWNLOAD_MODELS)
```

<a id="stage-the-tutorial-inputs-and-workflows"></a>

### 暂存教程输入和工作流程

**第 2 部分** 需要使用本教程进行版本控制的两组小文件：

- **输入照片**：三个 768x768 PNG（`apple`、`banana`、`pineapple`）。
- **工作流程定义**：两个 ComfyUI API 格式的 JSON 文件，两个工作流程各一个。

> 这些照片源自 [Wikimedia Commons]（https://commons.wikimedia.org/wiki/Main_Page);，请参阅 [输入照片自述文件](https://github.com/ROCm/ComfyUI/blob/amd-integration/examples/assets/README.md) 了解适用的图像许可和归属。

这些文件与 `utils_comfyui.py` 帮助程序模块一起在 `examples/` 下的 [ROCm/ComfyUI](https://github.com/ROCm/ComfyUI/tree/amd-integration/examples) 存储库中进行版本控制，而不是在此Notebook存储库中。下一个单元下载此Notebook旁边的帮助程序模块（如果缺少），并将资产和工作流程暂存在Notebook的 `assets/` 和 `workflows/` 目录下。如果您正在运行已包含这些文件的 ComfyUI 结帐程序，它会在本地找到它们；否则它会从 `RAW_BASE_URL` 下载它们。

```python
import urllib.request

# This tutorial's helper module and supporting files (the workflow JSONs and
# input photos) are versioned in the ROCm/ComfyUI repo under examples/, not in
# this notebook repo. RAW_BASE_URL points there; the helper module is fetched next
# to this notebook if it is not already present (a no-op inside a checkout that
# already ships it).
RAW_BASE_URL = "https://raw.githubusercontent.com/ROCm/ComfyUI/amd-integration/examples"

if not (NOTEBOOK_DIR / "utils_comfyui.py").is_file():
    print("Fetching utils_comfyui.py from ROCm/ComfyUI ...")
    urllib.request.urlretrieve(f"{RAW_BASE_URL}/utils_comfyui.py",
                               str(NOTEBOOK_DIR / "utils_comfyui.py"))

from utils_comfyui import stage_files

ASSET_FILES = [
    "amd_comfyui_tutorial_apple_seed.png",
    "amd_comfyui_tutorial_banana_seed.png",
    "amd_comfyui_tutorial_pineapple_seed.png",
]
WORKFLOW_FILES = [
    "hunyuan3d_v2_1_image_to_3d.json",
    "wan2_2_5b_image_to_video.json",
]

print("Input photographs:")
stage_files(ASSET_FILES, ASSETS_DIR,
            [NOTEBOOK_DIR / "assets"], f"{RAW_BASE_URL}/assets")
print("Workflow definitions:")
stage_files(WORKFLOW_FILES, WORKFLOWS_DIR,
            [NOTEBOOK_DIR / "workflows"], f"{RAW_BASE_URL}/workflows")

missing = [f for f in ASSET_FILES if not (ASSETS_DIR / f).is_file()]
missing += [f for f in WORKFLOW_FILES if not (WORKFLOWS_DIR / f).is_file()]
assert not missing, f"Could not stage {missing}. Set RAW_BASE_URL or place the files manually."
print("All inputs and workflows are in place.")
```

<a id="launching-the-comfyui-server-in-the-background"></a>

### 在后台启动 ComfyUI 服务器

ComfyUI 作为长期 HTTP 服务器运行。下一个单元通过在后台运行 `python $COMFYUI_PATH/main.py --listen --port 8188` 来启动服务器，捕获其 PID，并等待 `/system_stats` 端点返回 200（表示“准备好接受提示”）。如果服务器已经在 `127.0.0.1:8188` 上运行（例如，您在打开 Jupyter 之前手动启动了它），单元会检测到这一点，并且这将变为无操作。

PID 存储在 `server_pid` 中。Notebook的最后一个单元将终止该过程。

```python
from utils_comfyui import start_comfyui_server

SERVER_PORT = int(os.environ.get("COMFYUI_PORT_HOST", "8188"))
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"

server_pid = start_comfyui_server(
    COMFYUI_PATH, SERVER_URL, SERVER_PORT, OUTPUTS_DIR / "comfyui_server.log"
)
```

<a id="the-comfyui-http-api-and-a-tiny-client"></a>

### ComfyUI HTTP API 和小型客户端

我们通过三个端点驱动服务器：

- `POST /prompt` 提交工作流程（返回 `prompt_id`）。
- `GET /history/<prompt_id>` 检查运行是否已完成。
- `GET /view` 下载生成的文件。

为了保持此Notebook的可读性，包装这些调用的微型客户端位于配套模块 [`utils_comfyui.py`](https://github.com/ROCm/ComfyUI/blob/amd-integration/examples/utils_comfyui.py) 中，该模块位于此Notebook旁边的上方单元格旁边。它的 `ComfyUIClient` 公开了 `queue_prompt`、`wait_for_completion` 和 `download_outputs`，以及链接所有三个的 `run_workflow` 便捷方法（提交 → 等待 → 下载）。

完整的 ComfyUI 服务器 API 记录在 [ComfyUI 服务器文档](https://docs.comfy.org/development/comfyui-server/comms_overview) 中。

```python
from utils_comfyui import ComfyUIClient

client = ComfyUIClient(SERVER_URL)
print("ComfyUI client ready ->", client.server_url)
```

<a id="part-2-generate-with-two-diffusion-workflows-on-real-photographs"></a>

## 第 2 部分：在真实照片上使用两个扩散工作流程生成

第 2 部分演示了两个 **独立** ComfyUI 生成工作流程，两者均通过 **第 1 部分 ** 中设置的 HTTP API 由 Python 驱动，并在同一组输入照片（苹果、香蕉、菠萝，每个 768x768）上运行。  

| |工作流程|输入 | AI输出|
|---|---|---|---|
|示例 A | **[Hunyuan3D v2.1 图像转 3D](https://huggingface.co/tencent/Hunyuan3D-2.1)**: 基于流的扩散变压器，`octree_resolution=4096` | 一张照片 | glTF (`.glb`) 格式的防水 3D 网格。几何体是 AI 输出。|
|示例 B | **[Wan2.2 5B图像转视频](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged)**（文本图像转视频变体）：5B参数视频扩散变压器|同一张照片加文本提示|一个MP4剪辑（81帧，24fps），每一帧由模型生成。|

第 2 部分末尾的可视化单元格布局了 3 行网格：

- **第 1 行（输入照片）：** 馈送到两个工作流程的源照片。
- **第 2 行（网格转盘）：** Hunyuan3D `.glb` 的确定性 CPU 渲染。这一行*不是*实际的网格输出，而是人工智能生成的网格的固定相机正交漫游。需要使用 3D 查看器来真正检查输出。
- **第 3 行（AI 相机轨道）：** Wan2.2 5B 图像到视频输出。这一行包含的剪辑是人工智能的实际产品。

输入的照片来源和许可证位于[输入照片自述文件](https://github.com/ROCm/ComfyUI/blob/amd-integration/examples/assets/README.md) 中。

<a id="example-a-hunyuan3d-v21-image-to-3d"></a>

### 示例 A：Hunyuan3D v2.1 图像转 3D

我们的第一个工作流程将每张照片变成 3D 网格。下面的单元格将三个输入图像暂存到 ComfyUI 的 `input/` 目录中，确保 Hunyuan3D 检查点 (~10 GB) 可用，并加载之前暂存的工作流程 JSON。

>**提示：** 如果模型 检查点 丢失，在上面的配置单元中设置 `DOWNLOAD_MODELS = True` 会导致模型 检查点 自动从 Hugging Face 下载到 `ComfyUI/models/checkpoints/`。使用脚本中的默认 `DOWNLOAD_MODELS = False`，单元会报告 检查点 丢失并跳过生成步骤；将标志翻转为 `True`，或者在运行工作流程之前手动将 检查点 下载到上述路径。

```python
from utils_comfyui import download_models

# Three real-world subjects. Photo credits/licences: examples/assets/README.md in ROCm/ComfyUI.
INPUTS = [
    ("apple",     ASSETS_DIR / "amd_comfyui_tutorial_apple_seed.png"),
    ("banana",    ASSETS_DIR / "amd_comfyui_tutorial_banana_seed.png"),
    ("pineapple", ASSETS_DIR / "amd_comfyui_tutorial_pineapple_seed.png"),
]

# Stage the photos where ComfyUI's LoadImage node looks for them.
COMFYUI_INPUT_DIR = COMFYUI_PATH / "input"
COMFYUI_INPUT_DIR.mkdir(exist_ok=True)
for _, asset in INPUTS:
    shutil.copy2(asset, COMFYUI_INPUT_DIR / asset.name)

# Checkpoint -> Hugging Face URL. Downloaded only when DOWNLOAD_MODELS is True.
HUNYUAN_CKPT = COMFYUI_PATH / "models" / "checkpoints" / "hunyuan_3d_v2.1.safetensors"
HUNYUAN_MODELS = {
    HUNYUAN_CKPT: "https://huggingface.co/Comfy-Org/hunyuan3D_2.1_repackaged/"
                  "resolve/main/hunyuan_3d_v2.1.safetensors",
}
HUNYUAN_READY = not download_models(HUNYUAN_MODELS, DOWNLOAD_MODELS)

with (WORKFLOWS_DIR / "hunyuan3d_v2_1_image_to_3d.json").open() as f:
    hunyuan_workflow = json.load(f)
print(f"Staged {len(INPUTS)} photos; workflow has {len(hunyuan_workflow)} nodes; checkpoint ready: {HUNYUAN_READY}")
```

<a id="run-the-workflow-over-each-input"></a>

### 对每个输入运行工作流程

以下单元作为“热身”运行一次工作流程（加载约 10 GB 的权重并编译扩散和变分自动编码器 (VAE) 内核），然后针对每个输入照片运行一次。每次调用都会交换 `LoadImage` 文件名和 `SaveGLB` 前缀，并碰撞 KSampler 种子，因此每次运行都是新鲜的，而不是从缓存返回。

```python
MESHES = {}
HUNYUAN_OUT = OUTPUTS_DIR / "hunyuan3d"


def customise_hunyuan(workflow, image_name, output_prefix):
    wf = deepcopy(workflow)
    wf["2"]["inputs"]["image"] = image_name
    wf["10"]["inputs"]["filename_prefix"] = output_prefix
    wf["7"]["inputs"]["seed"] = int(time.time() * 1_000_000) % (2**63 - 1)
    return wf


if HUNYUAN_READY:
    print("Warm-up (loads weights, compiles kernels)...")
    client.run_workflow(customise_hunyuan(hunyuan_workflow, INPUTS[0][1].name, "amd_tutorial/_warmup"),
                        HUNYUAN_OUT / "_warmup")
    for name, asset in INPUTS:
        wf = customise_hunyuan(hunyuan_workflow, asset.name, f"amd_tutorial/hunyuan3d_v2_1_{name}")
        saved, _ = client.run_workflow(wf, HUNYUAN_OUT / name)
        glbs = [p for p in saved if p.suffix.lower() == ".glb"]
        MESHES[name] = glbs[0] if glbs else None
        print(f"{name:<12} -> {MESHES[name].relative_to(NOTEBOOK_DIR) if MESHES[name] else '(no .glb)'}")
else:
    print("Skipping run; checkpoint not present.")
```

<a id="visualize-the-ai-generated-mesh-deterministic-turntable-render"></a>

### 可视化 AI 生成的网格：确定性转盘渲染

Hunyuan3D 的输出是磁盘上的 `.glb` 网格。为了在线检查它并确认模型构建了真实的 3D 对象，而不是平面贴纸，每个网格都被渲染为一个短的 MP4 转盘。

此渲染是**确定性的且仅依赖 CPU**（它“不是”AI 步骤）：它加载网格、对其进行抽取，并使用简单的兰伯特着色对 36 个旋转帧进行光栅化。这在 [`utils_comfyui.py`](https://github.com/ROCm/ComfyUI/blob/amd-integration/examples/utils_comfyui.py) 中实现为 `render_mesh_turntable`；下面的单元格只是在生成的每个网格上调用它。

```python
from utils_comfyui import render_mesh_turntable

TURNTABLE_OUT = OUTPUTS_DIR / "mesh_turntables"
TURNTABLE_VIDEOS = {}
for name, _ in INPUTS:
    glb = MESHES.get(name)
    if glb is None or not glb.is_file():
        print(f"{name:<12} (skip) - no mesh available")
        continue
    TURNTABLE_VIDEOS[name] = render_mesh_turntable(glb, TURNTABLE_OUT / f"{name}.mp4")
    print(f"{name:<12} -> {TURNTABLE_VIDEOS[name].relative_to(NOTEBOOK_DIR)}")
```

<a id="example-b-wan22-5b-image-to-video"></a>

### 示例 B：Wan2.2 5B 图像到视频

第二个工作流程将每张照片动画化为短视频（81 帧，24 fps），其中每一帧都是由扩散模型生成的 - 一个真正的 AI 相机轨道，而不是任何 3D 网格的光栅化。

下一个单元格包含每个受试者自己的积极提示，您可以在重新运行生成之前对其进行编辑。提示故意强调 *3D 相机运动*（相对于平面旋转），因为 Wan2.2 5B 在剪切式照片（例如纯白色背景上的水果）上的视差先验较弱，并且除非提示主动要求 3D 弧线，否则将默认旋转图像平面。负提示可防止在提示工程过程中观察到的故障模式（多个或去皮的水果、平面旋转、框架中的手指）。

```python
# Per-subject positive prompts for Wan2.2 5B I2V, keyed by INPUTS name.
# Edit these and re-run the workflow cell to experiment.
SUBJECT_PROMPTS: dict[str, str] = {
    "apple": (
        "Smooth slow 360 degree camera orbit around a single round red apple "
        "on a clean white background. The camera moves steadily and "
        "continuously around the apple in a wide 3D arc, revealing the back "
        "side of the apple. The apple itself is a single fruit and stays "
        "centred. Photorealistic studio product photography, sharp focus, "
        "soft natural lighting, shallow depth of field. No people, no hands, "
        "no text."
    ),
    "banana": (
        "Smooth slow 360 degree camera orbit around exactly one single yellow "
        "banana on a clean white background. There is only one banana - one "
        "single piece of fruit - lying on its side, stationary, intact, with "
        "the curve clearly visible. The camera arcs around the banana to "
        "reveal the back side of the curve in 3D. Photorealistic studio "
        "product photography, sharp focus, soft natural lighting."
    ),
    "pineapple": (
        "Smooth slow 360 degree camera orbit around a single tall pineapple "
        "with green crown leaves on a clean white background. The camera "
        "moves steadily and continuously around the pineapple in a wide 3D "
        "arc, revealing the back side of the pineapple body and the back of "
        "the crown leaves. The pineapple itself is a single fruit and stays "
        "centred. Photorealistic studio product photography, sharp focus, "
        "soft natural lighting. No people, no hands, no text."
    ),
}

NEGATIVE_PROMPT = (
    "multiple bananas, banana hand, banana cluster, multiple fruits, fruit "
    "cluster, duplicates, two apples, two bananas, two pineapples, sliced, "
    "peeled, cut open, chopped, broken fruit, mashed, deformed fruit, "
    "flat 2D rotation, planar rotation, sliding image, no parallax, "
    "human, person, hands, fingers, body, face, arms, lamp, text, watermark, "
    "logo, low quality, blurry, jpeg artifacts, distorted, deformed, extra "
    "limbs, motion blur, tilted, falling, flying, cartoon"
)

print(f"Defined positive prompts for {len(SUBJECT_PROMPTS)} subjects.")
print(f"Negative prompt: {len(NEGATIVE_PROMPT)} chars.")
```

定义提示后，下一个单元格将确保 Wan2.2 检查点（扩散模型、文本编码器和 VAE 的约 17 GB）可用，并准备每个主题的工作流程定制器。与示例 A 一样，仅当 `DOWNLOAD_MODELS = True` 时才从 Hugging Face 下载缺失的检查点；否则，单元将它们报告为丢失，并且跳过视频生成步骤。

```python
# Three checkpoints -> their Hugging Face URLs (Comfy-Org repackaged split files).
WAN_BASE = "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files"
WAN_MODELS = {
    COMFYUI_PATH / "models" / "diffusion_models" / "wan2.2_ti2v_5B_fp16.safetensors":
        f"{WAN_BASE}/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
    COMFYUI_PATH / "models" / "text_encoders" / "umt5_xxl_fp8_e4m3fn_scaled.safetensors":
        f"{WAN_BASE}/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    COMFYUI_PATH / "models" / "vae" / "wan2.2_vae.safetensors":
        f"{WAN_BASE}/vae/wan2.2_vae.safetensors",
}
WAN_I2V_READY = not download_models(WAN_MODELS, DOWNLOAD_MODELS)

with (WORKFLOWS_DIR / "wan2_2_5b_image_to_video.json").open() as f:
    wan_workflow = json.load(f)


def customise_wan(workflow, image_name, output_prefix, positive, negative=NEGATIVE_PROMPT):
    wf = deepcopy(workflow)
    wf["56"]["inputs"]["image"] = image_name
    wf["58"]["inputs"]["filename_prefix"] = output_prefix
    wf["3"]["inputs"]["seed"] = int(time.time() * 1_000_000) % (2**63 - 1)
    wf["6"]["inputs"]["text"] = positive
    wf["7"]["inputs"]["text"] = negative
    return wf


print(f"Loaded Wan2.2 workflow ({len(wan_workflow)} nodes); checkpoints ready: {WAN_I2V_READY}")
```

现在运行每个主题的工作流程：

```python
WAN_OUT = OUTPUTS_DIR / "wan2_2_5b_i2v"
WAN_VIDEOS = {}
if WAN_I2V_READY:
    print("Warm-up (loads weights, compiles kernels)...")
    client.run_workflow(
        customise_wan(wan_workflow, INPUTS[0][1].name, "amd_tutorial/_wan_warmup", SUBJECT_PROMPTS[INPUTS[0][0]]),
        WAN_OUT / "_warmup",
    )
    for name, asset in INPUTS:
        wf = customise_wan(wan_workflow, asset.name, f"amd_tutorial/wan22_5b_i2v_{name}", SUBJECT_PROMPTS[name])
        saved, _ = client.run_workflow(wf, WAN_OUT / name)
        mp4s = [p for p in saved if p.suffix.lower() == ".mp4"]
        WAN_VIDEOS[name] = mp4s[0] if mp4s else None
        print(f"{name:<12} -> {WAN_VIDEOS[name].relative_to(NOTEBOOK_DIR) if WAN_VIDEOS[name] else '(no .mp4)'}")
else:
    print("Skipping Wan2.2 5B I2V; checkpoints not present.")
```

###并排：输入照片、AI网格转盘、AI相机轨道

最后，将结果布置为网格 - 每个主题三行，一列：

- **第 1 行：输入照片。** 768×768 PNG 馈送到两个工作流程。
- **第 2 行：网格转盘。** Hunyuan3D `.glb` 的确定性 CPU 渲染（网格是 AI 输出；旋转是脚本化的）。
- **第 3 行：AI 相机轨道。** Wan2.2 5B 输出，其中每一帧都是由扩散模型生成的。

`result_grid_html`（在[`utils_comfyui.py`](https://github.com/ROCm/ComfyUI/blob/amd-integration/examples/utils_comfyui.py)中）将每个项目嵌入为base64，因此渲染的Notebook是独立的。

```python
from utils_comfyui import result_grid_html
from IPython.display import HTML, display

display(HTML(result_grid_html(INPUTS, MESHES, TURNTABLE_VIDEOS, WAN_VIDEOS)))
```

<a id="part-3-measure-throughput-on-your-own-hardware-optional"></a>

## 第 3 部分：在您自己的硬件上测量 吞吐量（可选）

这部分是**可选**。在下面的配置单元中设置 `RUN_BENCHMARK = True` 来运行它（默认值为 `False`，因此Notebook为初次阅读的读者提供了一个快速演示）。

**此实验的目标**是回答关于*您的*硬件的一个实际问题：*给定的工作流程从提交提示到保存输出需要多长时间？*该工具运行一次热身过程（加载权重并编译内核），然后进行几次定时迭代，然后报告**您自己的测量值的平均值和标准偏差**。时序逻辑位于 [`utils_comfyui.py`](https://github.com/ROCm/ComfyUI/blob/amd-integration/examples/utils_comfyui.py) 中的 `benchmark_workflow` 中。

此Notebook附带**没有预先记录的数字** - 下面的每个数字都是由您触发的运行生成的。如果您想将您的结果与 AMD 发布的参考数据进行比较，请参阅 AMD ROCm 博客 [*使用 ROCm 加速 AMD Instinct MI355X GPU 上的 ComfyUI 工作流程*](https://rocm.blogs.amd.com/artificial-intelligence/comfyui/README.html)。

默认情况下，线束重新运行 **第 2 部分** 中的 Hunyuan3D 工作流程。要对另一个工作流程（例如 FLUX.1-dev 或 Wan2.2 14B 图像到视频）进行基准测试，请将条目及其工作流程 JSON 和 检查点 路径附加到 `BENCH_SPECS`，该条目将被自动拾取，如果缺少任何引用的文件，则会跳过该条目。

```python
# Default OFF: flip to True to time iterations on your hardware.
RUN_BENCHMARK = False
ITERATIONS = 3   # timed passes per workflow, after one warm-up pass

# By default the benchmark is on the Hunyuan3D workflow from Part 2. Append more entries
# (same shape: key / workflow / checkpoints / image_input) to measure others.
BENCH_SPECS = [
    {
        "key": "hunyuan3d_v2_1",
        "workflow": WORKFLOWS_DIR / "hunyuan3d_v2_1_image_to_3d.json",
        "checkpoints": [HUNYUAN_CKPT],
        "image_input": INPUTS[0][1].name,
    },
]

BENCH_OUT = OUTPUTS_DIR / "part3_benchmarks"
print(f"Configured {len(BENCH_SPECS)} workflow(s); RUN_BENCHMARK={RUN_BENCHMARK}, ITERATIONS={ITERATIONS}.")
```

<a id="run-the-timed-iterations"></a>

### 运行定时迭代

下一个单元运行每个已配置的工作流程，其检查点存在：一次预热传递，然后是 `ITERATIONS` 定时传递。每次传递都会获得一个新鲜的种子，因此不会从缓存中提供任何内容。计时是围绕使用 `time.perf_counter()` 的 API 调用测量的挂钟秒数。

```python
from utils_comfyui import benchmark_workflow

BENCH_RESULTS = {}
if RUN_BENCHMARK:
    for spec in BENCH_SPECS:
        print(f"\n=== {spec['key']} ===")
        result = benchmark_workflow(
            client, spec["workflow"], spec["checkpoints"],
            spec["image_input"], ITERATIONS, BENCH_OUT / spec["key"],
        )
        if result is not None:
            BENCH_RESULTS[spec["key"]] = result
    print(f"\nFinished {len(BENCH_RESULTS)}/{len(BENCH_SPECS)} workflow(s).")
else:
    print("RUN_BENCHMARK=False; skipping Part 3 timed runs.")
    print("Set RUN_BENCHMARK=True in the configuration cell to enable.")
```

<a id="your-results"></a>

### 你的结果

下一个单元格生成一个表格，总结您在自己的 GPU 上运行的所有迭代。在设置 `RUN_BENCHMARK = True` 并重新运行上面的单元格之前，该表将保持为空。

```python
print(f"{'workflow':<20} {'iterations':>10} {'mean (s)':>12} {'std (s)':>12}")
print("-" * 56)
for spec in BENCH_SPECS:
    r = BENCH_RESULTS.get(spec["key"])
    if r is None:
        print(f"{spec['key']:<20} {'(skipped)':>10}")
    else:
        print(f"{spec['key']:<20} {len(r['times_s']):>10} {r['mean_s']:>12.2f} {r['std_s']:>12.2f}")

if not BENCH_RESULTS:
    print("\n(no local results - set RUN_BENCHMARK=True and re-run the runner cell)")
```

<a id="clean-up"></a>

### 清理

完成本教程后，终止已启动的后台 ComfyUI 服务器（如果重复使用预先存在的服务器，则无操作）。

```python
import signal

if server_pid is not None:
    try:
        os.kill(server_pid, signal.SIGTERM)
        print(f"Sent SIGTERM to ComfyUI server pid {server_pid}.")
    except ProcessLookupError:
        print(f"ComfyUI server pid {server_pid} already exited.")
else:
    print("No server was launched by this notebook; nothing to clean up.")
```

<a id="summary"></a>

## 总结

在本教程中，您在 AMD Instinct GPU 上从 Python 端对端驱动 ComfyUI 生成工作流程，无需浏览器：

- **设置** ComfyUI-on-ROCm 容器，验证 PyTorch + ROCm 运行时，在后台启动 ComfyUI 服务器，并加载一小组 HTTP API 帮助程序（`queue_prompt`、`wait_for_completion`、`download_outputs`、`run_workflow`）。
- **在相同的三张照片上生成**两种输出：使用**Hunyuan3D v2.1**（图像到3D）为每个主题生成一个3D网格，以及使用**Wan2.2 5B**（图像到视频）为每个主题生成一个简短的相机轨道视频，将其内联可视化为网格。
- **使用小型定时线束*（可选）*在您自己的硬件上测量**工作流程的端到端时间。

从这里，您可以尝试提交自己的工作流程 JSON（以 API 格式从 ComfyUI UI 导出任何工作流程）、交换不同的模型，或者通过相同的 HTTP API 将这些帮助程序连接到脚本或服务中。

自然而然的下一步就是扩展到单个 GPU 之外。虽然 ComfyUI 目前没有本机多 GPU 支持，但支持序列并行性、统一序列并行性 (USP) 和 PipeFusion 的社区工具（例如 [xDiT](https://github.com/xdit-project/xDiT)）是一个有希望的起点。

<a id="further-reading"></a>

### 进一步阅读

- [配套教程：AMD Radeon GPU 上的 ComfyUI（浏览器 UI）](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/t2v_comfyui_radeon.html)                                                         
- [配套博客：在 AMD Instinct MI355X 上加速 ComfyUI](https://rocm.blogs.amd.com/artificial-intelligence/comfyui/README.html)                                                                                              
- [ComfyUI 项目和文档](https://docs.comfy.org/)                                                                                                                                                                                
- [ComfyUI HTTP/WebSocket API 概述](https://docs.comfy.org/development/comfyui-server/comms_overview)                                                                                                                            
- [ROCm人工智能开发者中心](https://rocm.docs.amd.com/projects/ai-developer-hub)                                                                                                                                                       
- [混源3D v2.1](https://huggingface.co/tencent/Hunyuan3D-2.1)                                                                                                                                                                     
- [Wan2.2（Comfy-Org 重新包装）](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged)
