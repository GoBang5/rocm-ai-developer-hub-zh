> 原文作者：Ning Zhang  
> [AMD 官方教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/SpecForge_SGlang.html) · [固定版本源码](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/pretrain/SpecForge_SGlang.ipynb)  
> 上游版本：`0071bfd4bd3bd9525405c09a6ae69c983b09312a` · 同步日期：2026-10-12  
> 翻译状态：机器初译，待逐篇人工校对；代码与命令保持原文，未在 AMD GPU 上运行验证。

<!-- cell:1 -->

<a id="specforge-training-speculative-decoding-models-with-sglang"></a>

# 使用 SpecForge 训练投机解码草稿模型
**作者**：Ning Zhang
**知识水平**：中级

投机解码 是一种强大的技术，可以改进内存绑定 LLM 推理 中的 Token 间延迟，该技术需要并行运行两个模型。它们是*目标模型*（人工智能应用程序的主要LLM模型）和*小型草稿模型*（一个较小的、轻量级的LLM，与其一起运行，以帮助加快主要LLM的推理）。小草稿模型在投机解码中很重要，因为它能否正确预测目标模型接受的代币是部署是否成功的关键因素。

[SpecForge](https://github.com/sgl-project/SpecForge) 是一个专门构建的生态系统，用于训练草稿模型，与 SGLang 原生集成，是一个开源应用程序，被大型模型系统组织 (LMSYS) 列为旗舰项目。本教程演示如何在配备 AMD Instinct™ MI300X GPU 的系统上运行 SpecForge 草稿模型 训练和 SGLang 投机解码 推理 以及经过训练的 草稿模型。  

<a id="prerequisites"></a>

## 前置条件
本教程是使用以下设置进行开发和测试的。

<a id="operating-system"></a>

### 操作系统
* **Ubuntu 22.04/24.04**：确保您的系统运行的是 Ubuntu 22.04 或 24.04。

<a id="hardware"></a>

### 硬件
* **AMD Instinct GPU**：本教程在具有八个 GPU 的 AMD Instinct MI300X GPU 节点上进行了测试。确保您使用的节点具有八个 AMD Instinct GPU 或具有 ROCm 支持的兼容硬件，并且您的系统满足[官方要求](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)。

<a id="software"></a>

### 软件
* **ROCm 6.3+**：本教程需要 ROCm 6.3 或更高版本。按照 [ROCm 安装指南](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html) 安装并验证 ROCm。安装后，使用 `amd-smi` 命令确认您的设置。 AMD 还提供预构建的 ROCm Docker 映像，例如 [ROCm PyTorch](https://hub.docker.com/r/rocm/pytorch)、[ROCm Ubuntu 22.04](https://hub.docker.com/r/rocm/dev-ubuntu-22.04) 和 [ROCm Ubuntu 24.04](https://hub.docker.com/r/rocm/dev-ubuntu-24.04)。您可以使用这些预构建的 Docker 映像来减少设置 ROCm 环境的工作量。

  **注意**：对于 ROCm 6.4 及更早版本，请使用 `rocm-smi` 命令而不是 `amd-smi`。

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
* 确保 Hugging Face API 令牌具有访问 [Meta Llama 检查点](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) 所需的权限和批准。

<a id="set-up-the-specforge-environment"></a>

## 设置 SpecForge 环境 

在本教程中，您将使用预构建的 ROCm PyTorch 映像作为示例。您还可以尝试其他 ROCm 图像。

<a id="step-1-launch-the-docker-image"></a>

### 第 1 步：启动 Docker 镜像
启动 Docker 容器。将 `/path/to/SpecForge_Project` 替换为主机上存储 SpecForge 代码和模型文件的目录的完整路径。本教程使用 [SGLang ROCm Docker 映像](https://hub.docker.com/r/lmsysorg/sglang/tags?name=rocm) 演示 SpecForge 草稿模型 训练和 SGLang 投机解码 推理。为了获得更好的模型训练性能，您还可以尝试[ROCm PyTorch Training Docker镜像](https://hub.docker.com/r/rocm/pytorch-training/)。

``` bash
docker run -it --rm \
  --network=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size 32G \
  -v /path/to/SpecForge_Project:/SpecForge \
  -w /SpecForge/ \
  lmsysorg/sglang:v0.4.10.post2-rocm630-mi30x
```
**注意**：该命令将当前目录挂载到容器中的`/SpecForge`目录中。确保在运行 Docker 命令之前将Notebook文件复制到此目录，或者在启动后上传到 Jupyter Notebook 环境。保存终端输出中提供的令牌或 URL，以便从 Web 浏览器访问Notebook。您可以从 [AI Developer Hub GitHub 存储库](https://github.com/ROCm/gpuaidev) 下载此Notebook。

<a id="step-2-install-and-launch-jupyter"></a>

### 步骤 2：安装并启动 Jupyter

在 Docker 容器内，使用以下命令安装 Jupyter：

``` bash
pip install jupyter
```

启动 Jupyter 服务器：

``` bash
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**注意**：在运行上述命令之前，请确保您的系统上尚未使用端口 `8888`。如果是，您可以通过将 `--port=8888` 替换为其他端口号来指定其他端口，例如 `--port=8890`。

<a id="step-3-install-specforge"></a>

### 步骤 3：安装 SpecForge

直接安装[官方SpecForge代码](https://github.com/sgl-project/SpecForge)也会安装一些非AMD相关的GPU库，不推荐。为了避免这个问题，请选择正确的[提交](https://github.com/zhangnju/SpecForge)并进行一些修改。在 Docker 容器中运行的 Jupyter Notebook 中运行以下命令：

<details markdown="1">
<summary>查看英文原文 · 单元 1</summary>

# SpecForge: Training speculative decoding models with SGLang

**Author**: Ning Zhang  
**Knowledge level**: Intermediate

Speculative decoding is a powerful technique to improve inter-token latency in memory-bound LLM inference, which needs to run two models in parallel. These are the *target model* (the main LLM model for the AI application) and the *small draft model* (a smaller, lightweight LLM that runs alongside it to help speed up inference for the main LLM). The small draft model is important in speculative decoding because its ability to correctly predict the tokens for target model acceptance is the key factor in whether the deployment will be successful or not.

[SpecForge](https://github.com/sgl-project/SpecForge), a purpose-built ecosystem for training draft models that integrate natively with SGLang, is an open-source application that is listed as a flagship project by the Large Model Systems Organization (LMSYS). This tutorial demonstrates how to run SpecForge draft model training and SGLang speculative decoding inference with the trained draft model on a system with an AMD Instinct™ MI300X GPU.  

## Prerequisites
This tutorial was developed and tested using the following setup.

### Operating system
* **Ubuntu 22.04/24.04**: Ensure your system is running Ubuntu 22.04 or 24.04.

### Hardware
* **AMD Instinct GPUs**: This tutorial was tested on an AMD Instinct MI300X GPU node with eight GPUs. Ensure you are using a node with eight AMD Instinct GPUs or compatible hardware with ROCm support and that your system meets [the official requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

### Software
* **ROCm 6.3+**:  This tutorial requires ROCm 6.3 or later. Install and verify ROCm by following the [ROCm install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html). After installation, confirm your setup using the `amd-smi` command. AMD also provides prebuilt ROCm Docker images, for example, [ROCm PyTorch](https://hub.docker.com/r/rocm/pytorch), [ROCm Ubuntu 22.04](https://hub.docker.com/r/rocm/dev-ubuntu-22.04), and [ROCm Ubuntu 24.04](https://hub.docker.com/r/rocm/dev-ubuntu-24.04). You can use these prebuilt Docker images to reduce the effort of setting up a ROCm environment.

  **Note**: For ROCm 6.4 and earlier, use the `rocm-smi` command instead of `amd-smi`.

* **Docker**: Ensure Docker is installed and configured correctly. Follow the Docker installation guide for your operating system.

   **Note**: Ensure the Docker permissions are correctly configured. To configure permissions to allow non-root access, run the following commands:

   ``` bash
   sudo usermod -aG docker $USER
   newgrp docker
   ```

   Verify Docker is working correctly with:

   ``` bash
   docker run hello-world
   ```

### Hugging Face API access

* Obtain an API token from [Hugging Face](https://huggingface.co) for downloading models.
* Ensure the Hugging Face API token has the necessary permissions and approval to access the [Meta Llama checkpoints](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).

## Set up the SpecForge environment 

In this tutorial, you will work on the prebuilt ROCm PyTorch image as an example. You can also try the other ROCm images.

### Step 1: Launch the Docker image 

Launch the Docker container. Replace `/path/to/SpecForge_Project` with the full path to the directory on your host machine where the SpecForge code and model files are stored. This tutorial uses the [SGLang ROCm Docker images](https://hub.docker.com/r/lmsysorg/sglang/tags?name=rocm) to demonstrate SpecForge draft model training and SGLang speculative decoding inference. For better model training performance, you can also try the [ROCm PyTorch Training Docker image](https://hub.docker.com/r/rocm/pytorch-training/).

``` bash
docker run -it --rm \
  --network=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size 32G \
  -v /path/to/SpecForge_Project:/SpecForge \
  -w /SpecForge/ \
  lmsysorg/sglang:v0.4.10.post2-rocm630-mi30x
```
**Note**: This command mounts the current directory to the `/SpecForge` directory in the container. Ensure the notebook file is either copied to this directory before running the Docker command or uploaded into the Jupyter Notebook environment after it starts. Save the token or URL provided in the terminal output to access the notebook from your web browser. You can download this notebook from the [AI Developer Hub GitHub repository](https://github.com/ROCm/gpuaidev).

### Step 2: Install and launch Jupyter

Inside the Docker container, install Jupyter using the following command:

``` bash
pip install jupyter
```

Start the Jupyter server:

``` bash
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**Note**: Ensure port `8888` is not already in use on your system before running the above command. If it is, you can specify a different port by replacing `--port=8888` with another port number, for example, `--port=8890`.

### Step 3: Install SpecForge

Directly installing the [official SpecForge code](https://github.com/sgl-project/SpecForge) also installs some non-AMD related GPU libraries, which is not recommended. To avoid this issue, pick up the correct [commit](https://github.com/zhangnju/SpecForge) and make some modifications. Run the following commands inside the Jupyter notebook running within the Docker container:

</details>

```python
%%bash
git clone https://github.com/zhangnju/SpecForge.git
pip install -v ./SpecForge
```

<!-- cell:3 -->

<a id="step-4-provide-your-hugging-face-token"></a>

### 第 4 步：提供您的 Hugging Face 令牌

您需要 Hugging Face API 令牌才能访问 Llama-3。在 [Hugging Face Tokens](https://huggingface.co/settings/tokens) 生成您的令牌并请求访问 [Llama-3 8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)。令牌通常以“hf_”开头。 

在 Jupyter Notebook中运行以下交互式块来设置令牌：

**注意**：取消选中“添加令牌作为 Git 凭证”选项。

<details markdown="1">
<summary>查看英文原文 · 单元 3</summary>

### Step 4: Provide your Hugging Face token

You'll require a Hugging Face API token to access Llama-3. Generate your token at [Hugging Face Tokens](https://huggingface.co/settings/tokens) and request access for [Llama-3 8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct). Tokens typically start with "hf_". 

Run the following interactive block in your Jupyter notebook to set up the token:

**Note**: Uncheck the "Add token as Git credential" option.

</details>

```python
from huggingface_hub import notebook_login

# Prompt the user to log in
notebook_login()
```

```python
from huggingface_hub import HfApi

try:
    api = HfApi()
    user_info = api.whoami()
    print(f"Token validated successfully! Logged in as: {user_info['name']}")
except Exception as e:
    print(f"Token validation failed. Error: {e}")
```

<!-- cell:6 -->

<a id="speculative-decoding-draft-model-training"></a>

## 投机解码 草稿模型 训练

SpecForge 是一个用于训练 投机解码 模型的框架，因此您可以将它们平滑地移植到 SGLang 服务框架，以加速您的 推理。它提供了两种训练草稿模型的方法：在线和离线训练。在线训练冻结 目标模型 并同时训练 草稿模型，动态生成辅助隐藏状态，需要多个 GPU 才能获得更好的性能。离线训练首先使用 目标模型 生成并保存隐藏状态，然后在单独的过程中训练 草稿模型。离线训练只需要一个GPU，因为它只需要容纳草稿模型，但需要大量的磁盘空间。例如，UltraChat 和 ShareGPT 数据集需要 12TB 的存储空间。由于AMD Instinct MI300X系统的磁盘大小，本教程使用在线培训来演示SpecForge。  

<a id="data-preparation"></a>

### 数据准备 

SpecForge 提供了一个脚本来准备用于 草稿模型 训练的 UltraChat (200k) 和 ShareGPT (120k) 数据集。以下命令将数据集处理为 `jsonl`（JSON Lines）文件，这是在线训练的原始数据集，并将它们放在 `cache/dataset/<dataset_name>` 目录中。要使用您自己的数据训练模型，请准备 `jsonl` 格式的数据集。该架构应如下所示：

```json
{
    "id": "xxxx",
    "conversations": [
        {
            "role": "user | assistant",
            "content": "The message content"
        }
    ],
}
```

本教程以 ShareGPT 和 UltraChat 数据集为例。

<details markdown="1">
<summary>查看英文原文 · 单元 6</summary>

## Speculative decoding draft model training

SpecForge is a framework for training speculative decoding models so you can smoothly port them over to the SGLang serving framework to speed up your inference. It offers two methods of training the draft model: online and offline training. Online training freezes the target model and training draft model at same time, which generates auxiliary hidden states on the fly and needs multiple GPUs to achieve better performance. Offline training generates and saves the hidden states using the target model first and then trains the draft model in a separate process. Offline training requires as little as one GPU because it only needs to accommodate the draft model, but it needs a huge amount of disk space. For example, the UltraChat and ShareGPT datasets would need 12TB of storage. Due to the disk size of the AMD Instinct MI300X system, this tutorial uses online training to demonstrate SpecForge.  

### Data preparation 

SpecForge provides a script to prepare the UltraChat (200k) and ShareGPT (120k) datasets for draft model training. The commands below process the datasets into `jsonl` (JSON Lines) files, which are the raw dataset for online training, and place them in the `cache/dataset/<dataset_name>` directory. To train the model using your own data, prepare the dataset in `jsonl` format. The schema should look like this:

```json
{
    "id": "xxxx",
    "conversations": [
        {
            "role": "user | assistant",
            "content": "The message content"
        }
    ],
}
```

This tutorial uses the ShareGPT and UltraChat datasets as examples.

</details>

```python
%%bash
# ultrachat
python SpecForge/scripts/prepare_data.py --dataset ultrachat

# sharegpt
python SpecForge/scripts/prepare_data.py --dataset sharegpt
```

<!-- cell:8 -->

<a id="draft-model-online-training"></a>

### 草稿模型 在线培训 

SpecForge 提供了示例脚本来为 Llama3/4、Qwen3 和其他流行的 LLM 模型训练 草稿模型。本教程使用 Llama-3 8B 模型来演示如何使用 SpecForge 运行在线训练。由于模型训练需要很长时间，因此您只会运行一个 epoch 的训练，这可能不会获得最佳的准确性。您可以根据需要更改以下命令中的训练选项。

<details markdown="1">
<summary>查看英文原文 · 单元 8</summary>

### Draft model online training 

SpecForge provides sample scripts to train the draft model for Llama3/4, Qwen3, and other popular LLM models. This tutorial uses the Llama-3 8B model to demonstrate how to run online training using SpecForge. Because model training takes a long time, you will only run one epoch of training, which might not have the best accuracy. You can change the training options in the command below as required.

</details>

```python
%%bash
# Llama-3 8B draft model training using 8 GPUs 
torchrun \
    --standalone \
    --nproc_per_node 8 \
    SpecForge/scripts/train_eagle3_online.py \
    --target-model-path meta-llama/Meta-Llama-3-8B-Instruct \
    --draft-model-config SpecForge/configs/llama3-8B-eagle3.json \
    --train-data-path SpecForge/cache/dataset/sharegpt.jsonl \
    --output-dir outputs/llama3-8b-eagle3 \
    --num-epochs 1 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template llama3 \
    --cache-dir cache
```

<!-- cell:10 -->

训练完成后，可以在`output-dir`目录下找到训练好的草稿模型文件。

##（可选）SGLang 投机解码 推理 

本节展示如何基于SGLang框架测试训练好的草稿模型。如果您已经熟悉 SGLang 投机解码 推理，则可以跳过本节。

<a id="optional-sglang-speculative-decoding-inference"></a>

### SGLang 投机解码 服务器

SGLang 推理 服务器通过以下选项支持 EAGLE 投机解码：
1) `speculative-draft-model-path`：草稿模型。此参数是必需的。
2) `speculative-num-steps`：指定自回归绘图的深度。该变量增加了投机范围，但存在拒绝级联的风险。默认值为 `5`。
3) `speculative-eagle-topk`：指定每步的分支因子。这提高了候选人的多样性。它会带来更高的接受率，但也会导致更高的内存/计算消耗。默认值为 `4`。
4) `speculative-num-draft-tokens`：最大并行验证能力。它会导致更深的树评估，但会导致更高的 GPU 内存使用量。默认值为 `8`。 

以下 SGLang 服务器命令中这些选项的设置旨在演示 投机解码 功能测试，而不是最大化性能。您可以在 [bench_speculative.py](https://github.com/sgl-project/sglang/blob/main/scripts/playground/bench_speculative.py) 中找到这些参数的最佳组合。

**注意**：从终端运行本节中的命令，而不是从Notebook代码单元运行。在 JupyterLab 中，使用 **Launcher** → **Terminal** （或 **File** → **New** → **Terminal**）打开终端。

<details markdown="1">
<summary>查看英文原文 · 单元 10</summary>

After the training is done, you can find the trained draft model file in the `output-dir` directory.

## (Optional) SGLang speculative decoding inference 

This section shows how to test the trained draft model based on the SGLang framework. If you are already familiar with SGLang speculative decoding inference, you can skip this section.

### SGLang speculative decoding server

The SGLang inference server supports EAGLE speculative decoding through the following options:
1) `speculative-draft-model-path`: Specifies the draft model. This parameter is required.
2) `speculative-num-steps`: Specifies the depth of auto-regressive drafting. This variable increases the speculation range but risks rejection cascades. The default is `5`.
3) `speculative-eagle-topk`: Specifies the branching factor per step. This improves candidate diversity. It leads to a higher acceptance rate but also causes higher memory/compute consumption. The default is `4`.
4) `speculative-num-draft-tokens`: Specifies the maximum parallel verification capacity. It causes deeper tree evaluation but leads to higher GPU memory usage. The default is `8`. 

The settings for these options in the following SGLang server command are intended to demonstrate the speculative decoding functional test and are not about maximizing performance. You can find the best combinations of these parameters in [bench_speculative.py](https://github.com/sgl-project/sglang/blob/main/scripts/playground/bench_speculative.py).

**Note**: Run the commands in this section from a terminal, not from the notebook code cells. In JupyterLab, open a terminal using **Launcher** → **Terminal** (or **File** → **New** → **Terminal**).

</details>

```python
%%bash
python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3-8B-Instruct \
         --speculative-algorithm EAGLE3 --speculative-draft-model-path ./outputs/llama3-8b-eagle3/epoch_0 \
         --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
         --mem-fraction-static 0.75 --cuda-graph-max-bs 2 --tp 8 --context-length 8192 --trust-remote-code \
         --host 0.0.0.0 --port 30000 --dtype bfloat16 --attention-backend triton
```

<!-- cell:12 -->

<a id="sglang-speculative-decoding-benchmarking"></a>

### SGLang 投机解码 基准测试 

SpecForge 为 投机解码 提供了多个基准测试脚本。启动 SGLang 推理 服务器后，如上所示，您可以运行这些基准测试脚本，以使用不同的测试数据集测试 投机解码 性能。

<details markdown="1">
<summary>查看英文原文 · 单元 12</summary>

### SGLang speculative decoding benchmarking 

SpecForge provides multiple benchmark scripts for speculative decoding. After launching the SGLang inference server, as shown above, you can run these benchmarking scripts to test speculative decoding performance with different test datasets.

</details>

```python
%%bash
# GSM8K
python SpecForge/benchmarks/run_gsm8k.py

# MATH-500
python SpecForge/benchmarks/run_math500.py

# MTBench
python SpecForge/benchmarks/run_mtbench.py

# HumanEval
python SpecForge/benchmarks/run_humaneval.py
```

<!-- cell:14 -->

<a id="summary"></a>

## 总结 
投机解码已成为加速LLM 推理的突破口。 SpecForge 专为 投机解码 草稿模型 训练而设计，并与 SGLang 推理 引擎紧密集成。本教程演示了如何在 AMD GPU 上使用 SGLang 运行 SpecForge 草稿模型 训练并测试模型。 SpecForge 本身在不断发展，AMD 正在与开源社区密切合作，以增强 AMD 平台的这项工作。希望本教程能够鼓励您在 AMD GPU 上尝试、测试 SpecForge 并为其做出贡献，并帮助塑造 AI 加速的未来。

<details markdown="1">
<summary>查看英文原文 · 单元 14</summary>

## Summary 
Speculative decoding has emerged as a breakthrough for accelerating LLM inference. SpecForge is designed for speculative decoding draft model training and is tightly integrated with the SGLang inference engine. This tutorial demonstrated how to run SpecForge draft model training and test the model using SGLang on AMD GPUs. SpecForge itself is constantly evolving, and AMD is working closely with the open-source community to enhance this work for AMD platforms. This tutorial will hopefully encourage you to try, test, and contribute to SpecForge on AMD GPUs and help shape the future of AI acceleration.

</details>
