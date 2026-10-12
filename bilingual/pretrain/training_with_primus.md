> 原文作者：Wei Cai  
> [AMD 官方教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/training_with_primus.html) · [固定版本源码](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/pretrain/training_with_primus.ipynb)  
> 上游版本：`0071bfd4bd3bd9525405c09a6ae69c983b09312a` · 同步日期：2026-10-12  
> 翻译状态：机器初译，待逐篇人工校对；代码与命令保持原文，未在 AMD GPU 上运行验证。

<!-- cell:1 -->

<a id="training-a-model-with-primus"></a>

# 使用 Primus 训练模型
**作者**：Wei Cai
**知识水平**：中级

本教程演示如何利用 Primus 在 AMD GPU 上使用 ROCm 预训练 [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) 大语言模型 (LLM)。 Primus 是适用于 AMD Instinct™ GPU 的统一且灵活的训练框架，旨在支持包括 Megatron 在内的多个训练引擎后端，以提供可扩展的高性能模型训练。性能加速由 [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo) 和 ROCm 库提供支持。

Primus with Megatron 旨在取代 [ROCm Megatron-LM 训练](https://rocm.docs.amd.com/en/docs-7.2.4/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html) 工作流程。要了解如何使用 Megatron 将工作负载从 Megatron-LM 迁移到 Primus，请参阅[将工作负载从 Megatron-LM 迁移到 Primus（Megatron 后端）](https://rocm.docs.amd.com/en/docs-7.2.4/how-to/rocm-for-ai/training/benchmark-docker/previous-versions/megatron-lm-primus-migration-guide.html)。

通过学习本教程，您将了解如何使用 Primus 设置和运行训练、利用其性能优化，并获得使用 Primus 简化模型训练工作流程的实践经验。

<details markdown="1">
<summary>查看英文原文 · 单元 1</summary>

# Training a model with Primus

**Author**: Wei Cai  
**Knowledge level**: Intermediate

This tutorial demonstrates how to pretrain the [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) large language model (LLM) using ROCm on AMD GPUs by leveraging Primus. Primus is a unified and flexible training framework for AMD Instinct™ GPUs that's designed to support multiple training engine backends, including Megatron, to deliver scalable, high-performance model training. Performance acceleration is powered by [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo) and the ROCm libraries.

Primus with Megatron is designed to replace the [ROCm Megatron-LM training](https://rocm.docs.amd.com/en/docs-7.2.4/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html) workflow. To learn how to migrate workloads from Megatron-LM to Primus with Megatron, see [Migrating workloads to Primus (Megatron backend) from Megatron-LM](https://rocm.docs.amd.com/en/docs-7.2.4/how-to/rocm-for-ai/training/benchmark-docker/previous-versions/megatron-lm-primus-migration-guide.html).

By following this tutorial, you'll learn how to set up and run training with Primus, take advantage of its performance optimizations, and gain practical experience in using Primus to streamline model training workflows.

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

* **AMD Instinct™ GPU**：本教程在 AMD Instinct MI300X GPU 的完整节点（八个 MI300X GPU）上进行了测试。确保您使用的是 AMD Instinct GPU 或支持 ROCm 的兼容硬件，并且您的系统满足[官方要求](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)。

<a id="software"></a>

### 软件

* **ROCm 7.0.0**：按照 [ROCm 安装指南](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html) 安装并验证 ROCm。安装后，使用以下命令确认您的设置：

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

<details markdown="1">
<summary>查看英文原文 · 单元 2</summary>

## Prerequisites

This tutorial was developed and tested using the following setup. 

### Operating system

* **Ubuntu 22.04**: Ensure your system is running Ubuntu 22.04.

### Hardware

* **AMD Instinct™ GPUs**: This tutorial was tested on a full node of AMD Instinct MI300X GPUs (eight MI300X GPUs). Ensure you are using AMD Instinct GPUs or compatible hardware with ROCm support and that your system meets the [official requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

### Software

* **ROCm 7.0.0**: Install and verify ROCm by following the [ROCm install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html). After installation, confirm your setup using:

    ``` bash
    amd-smi
    ```

    This command lists your AMD GPUs with relevant details.
    
    **Note**: For ROCm 6.4 and earlier, use the `rocm-smi` command instead.

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
docker pull rocm/primus:v25.9_gfx942
```

<a id="2-launch-the-docker-container"></a>

### 2.启动Docker容器

启动 Docker 容器并映射必要的目录。 

``` bash
docker run -it \
    --device /dev/dri \
    --device /dev/kfd \
    --device /dev/infiniband \
    --network host --ipc host \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    --shm-size 128G \
    --name primus_training_env \
    rocm/primus:v25.9_gfx942
```

**注意**：如果退出后需要返回到 `primus_training_env` 容器，请使用以下命令：

``` bash
docker start primus_training_env
docker exec -it primus_training_env bash
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

<details markdown="1">
<summary>查看英文原文 · 单元 4</summary>

## Set up the environment

Follow these steps to prepare the training environment.

### 1. Pull the Docker image

Ensure your system meets the [System Requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

Pull the Docker image required for this tutorial:

``` bash
docker pull rocm/primus:v25.9_gfx942
```

### 2. Launch the Docker container

Launch the Docker container and map the necessary directories. 

``` bash
docker run -it \
    --device /dev/dri \
    --device /dev/kfd \
    --device /dev/infiniband \
    --network host --ipc host \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    --shm-size 128G \
    --name primus_training_env \
    rocm/primus:v25.9_gfx942
```

**Note**: If you need to return to the `primus_training_env` container after exiting it, use these commands:

``` bash
docker start primus_training_env
docker exec -it primus_training_env bash
```

**Note**: Ensure the notebook file is either copied to `/workspace` directory or uploaded into the Jupyter Notebook environment after it starts. Save the token or URL provided in the terminal output to access the notebook from your web browser. You can download this notebook from the [AI Developer Hub GitHub repository](https://github.com/ROCm/gpuaidev).


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

</details>

<!-- cell:5 -->

<a id="edit-the-training-configuration"></a>

## 编辑训练配置

Primus 在 YAML 中为 `examples/megatron/configs` 中的每个模型定义了训练配置。

例如，要更新 Qwen2.5-7B 的训练参数，请更新 `examples/megatron/configs/primus_qwen2.5_7B-pretrain.yaml`。其他模型的 YAML 训练配置文件遵循此命名约定。

<details markdown="1">
<summary>查看英文原文 · 单元 5</summary>

## Edit the training configuration

Primus defines a training configuration in YAML for each model in `examples/megatron/configs`.

For example, to update training parameters for Qwen2.5-7B, update `examples/megatron/configs/primus_qwen2.5_7B-pretrain.yaml`. The YAML training configuration files for other models follow this naming convention.

</details>

<!-- cell:6 -->

<a id="dataset-options"></a>

### 数据集选项

您可以使用模拟数据或真实数据进行训练。

- 模拟数据可用于测试和验证。使用 `mock_data` 字段在模拟数据和真实数据之间切换。默认值为 `true`（启用）。

  ``` yaml
  mock_data: true
  ```

- 如果您使用的是真实数据集，请更新 `train_data_path` 字段以指向您的数据集位置。

  ``` yaml
  mock_data: false
  train_data_path: /path/to/your/dataset
  ```

确保可在 Docker 容器内访问这些文件。

<details markdown="1">
<summary>查看英文原文 · 单元 6</summary>

### Dataset options

You can use either mock data or real data for training.

- Mock data can be useful for testing and validation. Use the `mock_data` field to toggle between mock and real data. The default value is `true` (for enabled).

  ``` yaml
  mock_data: true
  ```

- If you’re using a real dataset, update the `train_data_path` field to point to your dataset location.

  ``` yaml
  mock_data: false
  train_data_path: /path/to/your/dataset
  ```

Ensure that the files are accessible inside the Docker container.

</details>

```python
mock_data = True   # Set to False if you want to use a real dataset
train_data_path = "/path/to/your/dataset"  # Only used if mock_data=False

# Conditional logic
if mock_data:
    print("✅ Using mock dataset for testing and validation.")
    # Insert mock dataset loading logic here
else:
    print(f"✅ Using real dataset from: {train_data_path}")
    # Example: verify the dataset path exists
    import os
    if os.path.exists(train_data_path):
        print("📂 Dataset found and accessible.")
        # Insert real dataset loading logic here
    else:
        print("❌ Dataset path not found! Please check your configuration.")
```

<!-- cell:8 -->

<a id="run-training"></a>

## 运行训练
使用以下示例命令设置环境、配置[关键选项](https://rocm.docs.amd.com/en/docs-7.2.4/how-to/rocm-for-ai/training/benchmark-docker/previous-versions/primus-megatron-v25.8.html?model=primus_pyt_megatron_lm_train_qwen2.5-7b)，并使用 Primus 和 Megatron 后端在 AMD Instinct GPU 上运行训练。

<details markdown="1">
<summary>查看英文原文 · 单元 8</summary>

## Run training
Use the following example commands to set up the environment, configure [key options](https://rocm.docs.amd.com/en/docs-7.2.4/how-to/rocm-for-ai/training/benchmark-docker/previous-versions/primus-megatron-v25.8.html?model=primus_pyt_megatron_lm_train_qwen2.5-7b), and run training on AMD Instinct GPUs using Primus with the Megatron backend.

</details>

<!-- cell:9 -->

<a id="single-node-training"></a>

### 单节点训练
要在单个节点上运行训练，请导航到 `/workspace/Primus` 并使用以下部分中的设置命令：

<details markdown="1">
<summary>查看英文原文 · 单元 9</summary>

### Single node training
To run training on a single node, navigate to `/workspace/Primus` and use the setup commands in the following section:

</details>

```python
%cd /workspace/Primus
```

<!-- cell:11 -->

<a id="1-set-environment-variables"></a>

#### 1.设置环境变量

运行以下命令设置并确认环境变量。

<details markdown="1">
<summary>查看英文原文 · 单元 11</summary>

#### 1. Set environment variables

Run the following commands to set up and confirm the environment variables.

</details>

```python
# Install dependencies
!pip install -r requirements.txt

# Set environment variables for current process
import os
os.environ["HSA_NO_SCRATCH_RECLAIM"] = "1"
os.environ["NVTE_CK_USES_BWD_V3"] = "1"

# (Optional) Print to confirm they are set
print("HSA_NO_SCRATCH_RECLAIM =", os.environ.get("HSA_NO_SCRATCH_RECLAIM"))
print("NVTE_CK_USES_BWD_V3 =", os.environ.get("NVTE_CK_USES_BWD_V3"))
```

<!-- cell:13 -->

设置完成后，运行适当的训练命令。以下运行命令是针对Qwen2.5-7B定制的。请参阅[支持的型号](https://rocm.docs.amd.com/projects/primus/en/latest/06-developer-guide/model-support-matrix.html) 以切换到其他可用型号。

<details markdown="1">
<summary>查看英文原文 · 单元 13</summary>

After setup is complete, run the appropriate training commands. The following run commands are tailored to Qwen2.5-7B. See [Supported models](https://rocm.docs.amd.com/projects/primus/en/latest/06-developer-guide/model-support-matrix.html) to switch to another available model.

</details>

<!-- cell:14 -->

<a id="2-show-the-training-configuration"></a>

#### 2.显示训练配置

<details markdown="1">
<summary>查看英文原文 · 单元 14</summary>

#### 2. Show the training configuration

</details>

````python
# Display the training config YAML file
import yaml
from IPython.display import Markdown, display

with open("examples/megatron/configs/qwen2.5_7B-pretrain.yaml", "r") as f:
    train_config = yaml.safe_load(f)

yaml_text = yaml.dump(
    train_config,
    sort_keys=False,
    default_flow_style=False
)

display(Markdown(f"```yaml\n{yaml_text}\n```"))
````

<!-- cell:16 -->

您可以直接在 YAML 文件中修改参数或使用 CLI 参数覆盖它们。

<details markdown="1">
<summary>查看英文原文 · 单元 16</summary>

You can modify the parameters directly in the YAML file or override them using CLI arguments.

</details>

<!-- cell:17 -->

<a id="3-launch-training"></a>

#### 3.启动培训

要在 Qwen2.5-7B 的单个节点上运行训练，请使用以下命令：

<details markdown="1">
<summary>查看英文原文 · 单元 17</summary>

#### 3. Launch training

To run training on a single node for Qwen2.5-7B, use the following command:

</details>

```python
import subprocess

# Choose training mode: "bf16" or "fp8"
mode = "bf16"  # change to "fp8" if you want FP8 training

if mode == "bf16":
    cmd = """
    EXP=examples/megatron/configs/qwen2.5_7B-pretrain.yaml \
    bash examples/run_pretrain.sh \
        --train_iters 50
    """
elif mode == "fp8":
    cmd = """
    EXP=examples/megatron/configs/qwen2.5_7B-pretrain.yaml \
    bash examples/run_pretrain.sh \
        --train_iters 50 \
        --fp8 hybrid
    """
else:
    raise ValueError("Unsupported mode. Use 'bf16' or 'fp8'.")

print("Running command:\n", cmd)
subprocess.run(cmd, shell=True, check=True)
```

<!-- cell:19 -->

<a id="key-training-options"></a>

### 关键培训选项

以下关键选项适用于上面显示的训练命令：

**fp8**

* `hybrid` 启用 `FP8` GEMM。

**use_torch_fsdp2**

* `use_torch_fsdp2: 1` 启用torch FSDP-v2。
* 如果启用 FSDP，请将 `use_distributed_optimizer` 和 `overlap_param_gather` 设置为 `false`。

**个人资料**

* 要启用 PyTorch 分析，请设置以下参数：

  ``` yaml
  profile: true
  use_pytorch_profiler: true
  profile_step_end: 7
  profile_step_start: 6
  ```

**train_iters**

* 迭代总数（默认为`50`）。

**模拟数据**

* 默认为 `True`。

**微批量大小**

* 微批大小。

**全局批量大小**

* 全球 批大小。

**重新计算粒度**

* 用于激活检查点。

**层数**

* 用于减少层数，例如使用代理模型。

<details markdown="1">
<summary>查看英文原文 · 单元 19</summary>

### Key training options

The following key options apply to the training commands shown above:

**fp8**

* `hybrid` enables `FP8` GEMMs.

**use_torch_fsdp2**

* `use_torch_fsdp2: 1` enables torch FSDP-v2.
* If FSDP is enabled, set `use_distributed_optimizer` and `overlap_param_gather` to `false`.

**profile**

* To enable PyTorch profiling, set these parameters:

  ``` yaml
  profile: true
  use_pytorch_profiler: true
  profile_step_end: 7
  profile_step_start: 6
  ```

**train_iters**

* The total number of iterations (default is `50`).

**mock_data**

* `True` by default.

**micro_batch_size**

* Micro batch size.

**global_batch_size**

* Global batch size.

**recompute_granularity**

* For activation checkpointing.

**num_layers**

* For using a reduced number of layers, for example, with proxy models.

</details>

<!-- cell:20 -->

<a id="further-reading"></a>

## 进一步阅读
- 有关 Primus 的介绍，请参阅 [Primus：AMD GPU 上大型模型的轻量级统一训练框架](https://rocm.blogs.amd.com/software-tools-optimization/primus/README.html)。
- 要了解如何结合使用 Primus 和 PyTorch 设置和运行训练，请参阅[使用 Primus 和 Pytorch TorchTitan 训练模型](https://rocm.docs.amd.com/projects/primus/en/latest/02-user-guide/torchtitan-training.html)。
- 要了解有关为 AMD Instinct MI300X 系列 GPU 配置系统的系统设置和管理实践的更多信息，请参阅[AMD Instinct MI300X 系统优化](https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/system-optimization/mi300x.html)。
- 有关使用 ROCm 进行 AI 的其他现成 Docker 映像的列表，请参阅 [AMD Infinity Hub](https://www.amd.com/en/developer/resources/infinity-hub.html#f-amd_hub_category=AI%20%26%20ML%20Models)。

<details markdown="1">
<summary>查看英文原文 · 单元 20</summary>

## Further reading
- For an introduction to Primus, see [Primus: A Lightweight, Unified Training Framework for Large Models on AMD GPUs](https://rocm.blogs.amd.com/software-tools-optimization/primus/README.html).
- To learn how to set up and run training with Primus in combination with PyTorch, see [Training a model with Primus and Pytorch TorchTitan](https://rocm.docs.amd.com/projects/primus/en/latest/02-user-guide/torchtitan-training.html).
- To learn more about system settings and management practices to configure your system for AMD Instinct MI300X series GPUs, see [AMD Instinct MI300X system optimization](https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/system-optimization/mi300x.html).
- For a list of other ready-made Docker images for AI with ROCm, see [AMD Infinity Hub](https://www.amd.com/en/developer/resources/infinity-hub.html#f-amd_hub_category=AI%20%26%20ML%20Models).

</details>
