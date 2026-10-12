> 原文作者：Shekhar Pandey and Liz Li  
> [AMD 官方教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/setup_tutorial.html) · [固定版本源码](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/pretrain/setup_tutorial.ipynb)  
> 上游版本：`0071bfd4bd3bd9525405c09a6ae69c983b09312a` · 同步日期：2026-10-12  
> 翻译状态：机器初译，待逐篇人工校对；代码与命令保持原文，未在 AMD GPU 上运行验证。

<!-- cell:1 -->

<a id="pretraining-with-megatron-lm"></a>

# Megatron-LM 训练环境配置
**作者**：Shekhar Pandey and Liz Li
**知识水平**：中级

本教程将引导您完成使用 AMD 的 ROCm Megatron-LM 框架预训练大型语言模型（例如 Llama-2 和 Llama-3）所需的设置和配置。

ROCm Megatron-LM 框架是 Megatron-LM 的专门分支，旨在在 AMD GPU 上高效训练大规模语言模型。通过利用 AMD Instinct™ MI300X 加速器，该框架提供：

* 增强人工智能工作负载的可扩展性和性能。
* 支持强大且流行的模型，例如 Llama-2、Llama-3 和 Llama-3.1。

主要特点包括：

* **变压器引擎 (TE)**：优化的变压器层实现。
* **Flash Attention 2.0**：更快且节省内存的注意力机制。
* **3D 并行性 (TP + SP + CP)**：张量、管道和序列并行性。
* **融合内核**：用于优化训练操作。
* **GEMM Tuning**：自动选择最佳矩阵乘法内核。

预先优化的模型包括：

* **Llama-2**：7B 和 70B
* **Llama-3 / Llama-3.1**：8B 和 70B

有关更多详细信息，请参阅 [Megatron-LM GitHub 存储库](https://github.com/ROCm/Megatron-LM)。

<details markdown="1">
<summary>查看英文原文 · 单元 1</summary>

# Pretraining with Megatron-LM

**Authors**: Shekhar Pandey and Liz Li  
**Knowledge level**: Intermediate

This tutorial walks you through the setup and configuration required to pretrain large-scale language models such as Llama-2 and Llama-3 using AMD’s ROCm Megatron-LM framework.

The ROCm Megatron-LM framework is a specialized fork of Megatron-LM designed to train large-scale language models efficiently on AMD GPUs. By leveraging AMD Instinct™ MI300X accelerators, this framework offers:

* Enhanced scalability and performance for AI workloads.
* Support for powerful and popular models such as Llama-2, Llama-3, and Llama-3.1.

Key features include:

* **Transformer Engine (TE)**: Optimized transformer layer implementations.
* **Flash Attention 2.0**: Faster and memory-efficient attention mechanisms.
* **3D Parallelism (TP + SP + CP)**: Tensor, pipeline, and sequence parallelism.
* **Fused Kernels**: For optimized training operations.
* **GEMM Tuning**: Automatically selects optimal matrix multiplication kernels.

Pre-optimized models include:

* **Llama-2**: 7B and 70B
* **Llama-3 / Llama-3.1**: 8B and 70B

See the [Megatron-LM GitHub repository](https://github.com/ROCm/Megatron-LM) for more details.

</details>

<!-- cell:2 -->

<a id="prerequisites"></a>

## 前置条件

本教程是使用以下设置进行开发和测试的。 

<a id="operating-system"></a>

### 操作系统

* **Ubuntu 22.04**：确保您的系统运行的是 Ubuntu 版本 22.04。

<a id="hardware"></a>

### 硬件

* **AMD Instinct GPU**：本教程在 AMD Instinct MI300X GPU 上进行了测试。确保您使用的是 AMD Instinct GPU 或支持 ROCm 的兼容硬件，并且您的系统满足[官方要求](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)。

<a id="software"></a>

### 软件

* **ROCm 6.2**：按照 [ROCm 安装指南](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html) 安装并验证 ROCm。安装后，使用以下命令确认您的设置：

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
- 从[Hugging Face](https://huggingface.co)获取API令牌用于下载模型。

<details markdown="1">
<summary>查看英文原文 · 单元 2</summary>

## Prerequisites

This tutorial was developed and tested using the following setup. 

### Operating system

* **Ubuntu 22.04**: Ensure your system is running Ubuntu version 22.04.

### Hardware

* **AMD Instinct GPUs**:  This tutorial was tested on an AMD Instinct MI300X GPU. Ensure you are using an AMD Instinct GPU or compatible hardware with ROCm support and that your system meets the [official requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

### Software

* **ROCm 6.2**: Install and verify ROCm by following the [ROCm install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html). After installation, confirm your setup using:

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


### Hugging Face API access
- Obtain an API token from [Hugging Face](https://huggingface.co) for downloading models.

</details>

<!-- cell:3 -->

<a id="rccl-bandwidth-test"></a>

## RCCL 带宽测试

ROCm 集体通信库 (RCCL) 是 GPU 标准集体通信例程的独立库。有关详细信息，请参阅 [RCCL 文档](https://rocm.docs.amd.com/projects/rccl/en/latest/index.html)。在开始预训练之前运行 RCCL 带宽测试有助于确保多 GPU 或多节点设置针对高效分布式训练进行优化。

要运行这些测试，请首先更改到 `rccl-tests` 目录。

``` bash
cd ~/rccl-tests
```

**测试1**

此测试在 8 个 GPU (`-g 8`) 上运行，扫描范围从 8 字节到 10 GB：

``` bash
./build/all_reduce_perf -b 8 -e 10G -f 2 -g 8
```

脚本完成后，输出的最后两行应类似于以下内容：

``` bash
<a id="out-of-bounds-values--0-ok"></a>

# Out of bounds values : 0 OK
<a id="avg-bus-bandwidth-----102428"></a>

# Avg bus bandwidth    : 102.428
```

`Avg bus bandwidth` 应该约为 100gb/秒，因为 MI300x 使用 8 个 PCIe 5.0 高性能网卡。

**测试2**

建议每个 GPU 使用一个 MPI 进程，并在单节点和多节点配置上使用 `-g 1` 以性能为导向的运行。在 8 个 GPU 上运行此测试的命令类似于：

``` shell
mpirun -np 8 --bind-to numa ./build/all_reduce_perf -b 8 -e 10G -f 2 -g 1
```

脚本完成后，输出的最后两行应类似于以下内容：

```bash
<a id="out-of-bounds-values--0-ok-1"></a>

# Out of bounds values : 0 OK
<a id="avg-bus-bandwidth-----110537"></a>

# Avg bus bandwidth    : 110.537
```

**测试3**

该测试在多个节点上运行。使用以下脚本在四个 MI300X GPU 节点上运行 RCCL 测试。根据需要修改路径和节点地址。

```shell
/home/$USER/ompi_for_gpu/ompi/bin/mpirun -np 32 -H tw022:8,tw024:8,tw010:8, tw015:8 \
--mca pml ucx \
--mca btl ^openib \
-x NCCL_SOCKET_IFNAME=ens41np0 \
-x NCCL_IB_HCA=rdma0:1,rdma1:1,rdma2:1,rdma3:1,rdma4:1,rdma5:1,rdma6:1,rdma7:1 \
-x NCCL_IB_GID_INDEX=3 \
-x NCCL_MIN_NCHANNELS=40 \
-x NCCL_DEBUG=version \
$HOME/rccl-tests/build/all_reduce_perf -b 8 -e 8g -f 2 -g 1
```

脚本完成后，输出的最后两行应类似于以下内容：

``` bash
<a id="out-of-bounds-values--0-ok-2"></a>

# Out of bounds values : 0 OK
<a id="avg-bus-bandwidth-----943037"></a>

# Avg bus bandwidth    : 94.3037
```

<details markdown="1">
<summary>查看英文原文 · 单元 3</summary>

## RCCL bandwidth test

ROCm Collective Communications Library (RCCL) is a standalone library of standard collective communication routines for GPUs. See the [RCCL documentation](https://rocm.docs.amd.com/projects/rccl/en/latest/index.html) for more information. Running a RCCL bandwidth test before starting pre-training helps ensure that the multi-GPU or multi-node setup is optimized for efficient distributed training.

To run these tests, first change to the `rccl-tests` directory.

``` bash
cd ~/rccl-tests
```

**Test 1**

This test runs on 8 GPUs (`-g 8`), scanning from 8 bytes to 10 GB:

``` bash
./build/all_reduce_perf -b 8 -e 10G -f 2 -g 8
```

After the script completes, the last two lines of the output should look similar to this:

``` bash
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 102.428
```

The `Avg bus bandwidth` should be approximately 100gb/sec because the MI300x uses 8x PCIe 5.0 high-performance networking cards.

**Test 2**

It's recommended to use one MPI process per GPU and `-g 1` for performance-oriented runs on both single-node and multi-node configurations. The command to run this test on 8 GPUs is similar to this:

``` shell
mpirun -np 8 --bind-to numa ./build/all_reduce_perf -b 8 -e 10G -f 2 -g 1
```

After the script completes, the last two lines of the output should look similar to this:

```bash
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 110.537
```

**Test 3**

This test runs on multiple nodes. Use the following script to run the RCCL test on four MI300X GPU nodes. Modify the paths and node addresses as required.

```shell
/home/$USER/ompi_for_gpu/ompi/bin/mpirun -np 32 -H tw022:8,tw024:8,tw010:8, tw015:8 \
--mca pml ucx \
--mca btl ^openib \
-x NCCL_SOCKET_IFNAME=ens41np0 \
-x NCCL_IB_HCA=rdma0:1,rdma1:1,rdma2:1,rdma3:1,rdma4:1,rdma5:1,rdma6:1,rdma7:1 \
-x NCCL_IB_GID_INDEX=3 \
-x NCCL_MIN_NCHANNELS=40 \
-x NCCL_DEBUG=version \
$HOME/rccl-tests/build/all_reduce_perf -b 8 -e 8g -f 2 -g 1
```

After the script completes, the last two lines of the output should look similar to this:

``` bash
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 94.3037
```

</details>

<!-- cell:4 -->

<a id="prepare-the-training-environment"></a>

## 准备训练环境

当您的系统满足 前置条件 要求后，请按照以下步骤设置培训环境。

<a id="1-system-configuration"></a>

### 1.系统配置

为了最大限度地提高性能，请遵循以下建议的步骤：

* **禁用 NUMA 自动平衡**

   禁用 NUMA 自动平衡可以提高应用程序性能。了解有关 NUMA 自动平衡效果的更多信息 [此处](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html#disable-numa-auto-balancing)。 

   检查当前的 NUMA 设置：

   ```bash
   cat /proc/sys/kernel/numa_balancing
   ```

   NUMA设置如下：

   * `0`：禁用
   * `1`：启用

   如有必要，请使用以下命令将其设置为 `0` 来禁用 NUMA 自动平衡：

   ```bash
   sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
   ```

<details markdown="1">
<summary>查看英文原文 · 单元 4</summary>

## Prepare the training environment

After your system meets the prerequisites, follow these steps to set up the training environment.

### 1. System configuration

To maximize performance, follow the recommended steps below:

* **Disable NUMA auto-balancing**

   Disabling NUMA auto-balancing can improve application performance. Learn more about the effects of NUMA auto-balancing [here](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html#disable-numa-auto-balancing). 

   Check the current NUMA setting:

   ```bash
   cat /proc/sys/kernel/numa_balancing
   ```

   The NUMA settings are as follows:

   * `0`: Disabled
   * `1`: Enabled

   If necessary, disable NUMA auto-balancing by setting it to `0` using this command:

   ```bash
   sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
   ```

</details>

<!-- cell:5 -->

<a id="2-pull-the-docker-image"></a>

### 2. 拉取 Docker 镜像

在终端中运行以下命令来提取预构建的 Docker 映像。 Docker 映像包含所有必要的依赖项，包括 PyTorch、PyTorch Lightning、ROCm 库和 Megatron-LM 实用程序。

``` bash
docker pull rocm/megatron-lm:24.12-dev
```

<details markdown="1">
<summary>查看英文原文 · 单元 5</summary>

### 2. Pull the Docker image

Run the following command in your terminal to pull the prebuilt Docker image. The Docker image contains all the necessary dependencies, including PyTorch, PyTorch Lightning, the ROCm libraries, and Megatron-LM utilities.

``` bash
docker pull rocm/megatron-lm:24.12-dev
```

</details>

<!-- cell:6 -->

<a id="3-launch-the-docker-container"></a>

### 3.启动 Docker 容器

在终端中运行以下命令以启动具有适当配置的 Docker 容器：

``` bash
docker run -it --rm \
  --device /dev/dri \
  --device /dev/kfd \
  --network host \
  --ipc host \
  --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --privileged \
  --name megatron-dev-env \
  -v $(pwd):/workspace \
  -w /workspace/notebooks \
  rocm/megatron-lm:24.12-dev \
  /bin/bash
```

**注意**：该命令将当前目录挂载到容器中的`/workspace`目录中。确保在运行 Docker 命令之前将Notebook文件复制到此目录，或者在启动后上传到 Jupyter Notebook 环境。保存终端输出中提供的令牌或 URL，以便从 Web 浏览器访问Notebook。您可以从 [AI Developer Hub GitHub 存储库](https://github.com/ROCm/gpuaidev-docs) 下载此Notebook。

<details markdown="1">
<summary>查看英文原文 · 单元 6</summary>

### 3. Launch the Docker container

Run the following command in your terminal to launch the Docker container with the appropriate configuration:

``` bash
docker run -it --rm \
  --device /dev/dri \
  --device /dev/kfd \
  --network host \
  --ipc host \
  --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --privileged \
  --name megatron-dev-env \
  -v $(pwd):/workspace \
  -w /workspace/notebooks \
  rocm/megatron-lm:24.12-dev \
  /bin/bash
```

**Note**: This command mounts the current directory to the `/workspace` directory in the container. Ensure the notebook file is either copied to this directory before running the Docker command or uploaded into the Jupyter Notebook environment after it starts. Save the token or URL provided in the terminal output to access the notebook from your web browser. You can download this notebook from the [AI Developer Hub GitHub repository](https://github.com/ROCm/gpuaidev-docs).

</details>

<!-- cell:7 -->

<a id="4-install-jupyter-and-start-the-server"></a>

### 4.安装Jupyter并启动服务器

在 Docker 容器内，使用以下命令安装 Jupyter：

``` bash
pip install jupyter
```

启动 Jupyter 服务器：

``` bash
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```
**注意**：在运行上述命令之前，请确保端口 `8888` 尚未在您的系统上使用。如果是，请通过将 `--port=8888` 替换为其他端口号来指定其他端口，例如 `--port=8890`。

<details markdown="1">
<summary>查看英文原文 · 单元 7</summary>

### 4. Install Jupyter and start the server

Inside the Docker container, install Jupyter using the following command:

``` bash
pip install jupyter
```

Start the Jupyter server:

``` bash
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```
**Note**: Ensure that port `8888` is not already in use on your system before running the above command. If it is, specify a different port by replacing `--port=8888` with another port number, for example, `--port=8890`.

</details>

<!-- cell:8 -->

<a id="5-clone-the-megatron-lm-repository"></a>

### 5. 克隆 Megatron-LM 存储库
在 Docker 容器内运行以下命令来克隆 Megatron-LM 存储库并导航到经过验证的提交：

<details markdown="1">
<summary>查看英文原文 · 单元 8</summary>

### 5. Clone the Megatron-LM repository
Run the following commands inside the Docker container to clone the Megatron-LM repository and navigate to the validated commit:

</details>

```python
# Clone the Megatron-LM repository and navigate to the validated commit
!git clone https://github.com/ROCm/Megatron-LM && cd Megatron-LM && git checkout bb93ccbfeae6363c67b361a97a27c74ab86e7e92
```

<!-- cell:10 -->

<a id="preparing-the-training-dataset"></a>

## 准备训练数据集

虽然下一个教程使用“模拟数据”来简化教程，但本节将介绍一个示例，展示如果没有预处理数据，如何预处理数据以进行训练。本部分使用 [BookCorpus 数据集](https://huggingface.co/datasets/bookcorpus/bookcorpus)，它是用于训练语言模型的书籍集合。它包含多样化且连续的文本段落，使其适合 预训练 任务。

在本教程中，您将：

* 下载并检查 BookCorpus 数据集。
* 将其转换为 JSONL 格式。
* 下载必要的分词器文件（`vocab.json` 和 `merges.txt`）。
* 预处理数据，以便使用 Megatron-LM 训练 大语言模型。

<details markdown="1">
<summary>查看英文原文 · 单元 10</summary>

## Preparing the training dataset

While the next tutorial uses "mock data" to simplify the tutorial, this section covers an example showing how to preprocess your data for training if you don’t have preprocessed data. This section uses the [BookCorpus dataset](https://huggingface.co/datasets/bookcorpus/bookcorpus), which is a collection of books that has been used for training language models. It contains diverse and continuous text passages, making it suitable for pretraining tasks.

In this tutorial, you will:

* Download and inspect the BookCorpus dataset.
* Convert it to JSONL format.
* Download the necessary tokenizer files (`vocab.json` and `merges.txt`).
* Preprocess the data for training a large language model with Megatron-LM.

</details>

<!-- cell:11 -->

<a id="1-download-and-inspect-the-bookcorpus-dataset"></a>

### 1. 下载并检查 BookCorpus 数据集

使用 Hugging Face 数据集库下载 BookCorpus 数据集。此步骤可确保您能够访问预处理所需的原始数据。

<details markdown="1">
<summary>查看英文原文 · 单元 11</summary>

### 1. Download and inspect the BookCorpus dataset

Use the Hugging Face datasets library to download the BookCorpus dataset. This step ensures you have access to the raw data needed for preprocessing.

</details>

```python
from datasets import load_dataset

# Load BookCorpus dataset
dataset = load_dataset("bookcorpus/bookcorpus", trust_remote_code=True, split="train")

# Inspect the dataset
print("Dataset Structure:", dataset)
print("Sample Data:", dataset[0])  # Access the first record
```

<!-- cell:13 -->

<a id="2-convert-to-the-jsonl-format"></a>

### 2.转换为JSONL格式

Megatron-LM 的预处理脚本要求输入采用 JSONL 格式，其中每一行将文档表示为 JSON 对象。此步骤将数据集转换为所需的格式。

<details markdown="1">
<summary>查看英文原文 · 单元 13</summary>

### 2. Convert to the JSONL format

Megatron-LM's preprocessing script requires that the input be in JSONL format, where each line represents a document as a JSON object. This step converts the dataset into the required format.

</details>

```python
import json
from tqdm import tqdm  # Import tqdm for progress bar

output_file = "bookcorpus.jsonl"

# Open the output file
with open(output_file, "w") as f:
    # Use tqdm to display progress
    for record in tqdm(dataset, desc="Saving dataset to JSONL", unit="record"):
        json.dump({"text": record["text"]}, f)
        f.write("\n")

print(f"Dataset saved to {output_file}")
```

<!-- cell:15 -->

在进行下一步之前，请确保数据集已正确转换为 JSONL 格式。检查前几行以确认结构。

<details markdown="1">
<summary>查看英文原文 · 单元 15</summary>

Before moving to the next step, ensure that the dataset is correctly converted to JSONL format. Inspect the first few lines to confirm the structure.

</details>

```python
# Inspect the first few lines of the JSONL file
with open(output_file, "r") as f:
    for i in range(5):  # Print the first 5 lines
        print(json.loads(f.readline()))
```

<!-- cell:17 -->

<a id="3-download-the-tokenizer-files"></a>

### 3.下载分词器文件

决定您将使用哪个标记器进行训练。例如，您可以选择 GPT-2 字节对编码 (BPE) 标记生成器。 Megatron 提供的预处理工具需要为每个分词器提供一些额外的文件。对于 BPE，请下载以下两个定义标记器规则的文件：

* `vocab.json`：将令牌映射到唯一 ID。
* `merges.txt`：指定子字单元如何组合成标记。

下载这些文件以正确标记数据集。

<details markdown="1">
<summary>查看英文原文 · 单元 17</summary>

### 3. Download the tokenizer files

Decide which tokenizer you are going to use for your training. As an example, you might pick the GPT-2 Byte Pair Encoding (BPE) tokenizer. The preprocessing tool provided by Megatron requires some additional files for each tokenizer. In the case of BPE, download the following two files that define the tokenizer rules:

* `vocab.json`: Maps the tokens to unique IDs.
* `merges.txt`: Specifies how subword units are combined into tokens.

Download these files to tokenize the dataset correctly.

</details>

```python
# Download vocab.json and merges.txt for GPT-2 tokenizer
!wget https://huggingface.co/gpt2/resolve/main/vocab.json -O vocab.json
!wget https://huggingface.co/gpt2/resolve/main/merges.txt -O merges.txt
```

<!-- cell:19 -->

<a id="4-preprocess-the-data"></a>

### 4. 预处理数据

此步骤对数据集进行标记，并将其转换为适合使用 Megatron-LM 训练 大语言模型 的二进制文件和索引文件。将 Megatron-LM 预处理脚本与转换后的 JSONL 数据集和标记生成器文件结合使用。

<details markdown="1">
<summary>查看英文原文 · 单元 19</summary>

### 4. Preprocess the data

This step tokenizes the dataset and converts it into binary and index files suitable for training a large language model using Megatron-LM. Use the Megatron-LM preprocessing script with the converted JSONL dataset and tokenizer files.

</details>

```python
!mkdir -p output
!python Megatron/tools/preprocess_data.py \
    --input bookcorpus.jsonl \
    --json-keys text \
    --output-prefix output/bookcorpus \
    --tokenizer-type GPT2BPETokenizer \
    --vocab-file vocab.json \
    --merge-file merges.txt \
    --workers 4 \
    --append-eod \
    --partitions 2 \
    --log-interval 1000000 \
    --split-sentences
```

<!-- cell:21 -->

**注意：** 您可能需要根据您选择的分词器和数据集修改这些参数。 

现在检查预处理期间生成的输出文件，以确保所有内容均已正确处理并准备好进行训练。

<details markdown="1">
<summary>查看英文原文 · 单元 21</summary>

**Note:** You might need to modify these parameters based on the tokenizer you pick and your dataset. 

Now check the output files generated during preprocessing to ensure everything is processed correctly and ready for training.

</details>

```python
# List output files
!ls output/
```

<!-- cell:23 -->

<a id="5-configure-the-network-interfaces-only-applicable-for-multi-node-training"></a>

### 5.配置网络接口（仅适用于多节点训练）

您必须设置 `NCCL_SOCKET_IFNAME` 和 `GLOO_SOCKET_IFNAME` 变量。虽然使用 `export` 命令可以轻松完成此操作，但该Notebook可以自动设置网络接口变量。

首先，通过运行以下命令安装 `iproute2` 软件包：

<details markdown="1">
<summary>查看英文原文 · 单元 23</summary>

### 5. Configure the network interfaces (only applicable for multi-node training)

You must set the `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` variables. While this is easily accomplished using the `export` command, this notebook can automatically set the network interface variables.

First, install the `iproute2` package by running the following command:

</details>

```python
!apt install -y iproute2
```

<!-- cell:25 -->

然后运行以下命令自动检测活动网络接口并根据第一个可用接口设置环境变量：

<details markdown="1">
<summary>查看英文原文 · 单元 25</summary>

Then run the following commands to automatically detect the active network interfaces and set the environment variables based on the first available interface:

</details>

```python
import os
import subprocess

# Detect the active network interface
try:
    result = subprocess.run(
        "ip -o link show | awk '{print $2, $9}' | grep 'UP' | awk '{print $1}' | sed 's/://g' | head -n 1",
        shell=True,
        check=True,
        capture_output=True,
        text=True
    )
    active_interface = result.stdout.strip()

    # Set environment variables
    os.environ['NCCL_SOCKET_IFNAME'] = active_interface
    os.environ['GLOO_SOCKET_IFNAME'] = active_interface

    # Verify the variables
    print(f"NCCL_SOCKET_IFNAME is set to: {os.environ['NCCL_SOCKET_IFNAME']}")
    print(f"GLOO_SOCKET_IFNAME is set to: {os.environ['GLOO_SOCKET_IFNAME']}")

except subprocess.CalledProcessError as e:
    print(f"Error detecting network interface: {e.stderr}")
```

<!-- cell:27 -->

运行命令后，使用以下命令验证活动网络接口：

<details markdown="1">
<summary>查看英文原文 · 单元 27</summary>

After running the commands, verify the active network interface using the following command:

</details>

```python
!ip a
```

<!-- cell:29 -->

确保检测到的接口与系统的活动网络接口匹配。如有必要，请修改上面的脚本或手动设置 `NCCL_SOCKET_IFNAME` 和 `GLOO_SOCKET_IFNAME` 变量。您可以使用以下指令手动设置这些变量：

``` bash
export NCCL_SOCKET_IFNAME=<network_interface>
export GLOO_SOCKET_IFNAME=<network_interface>
```

<details markdown="1">
<summary>查看英文原文 · 单元 29</summary>

Ensure the detected interface matches your system's active network interface. If necessary, modify the script above or manually set the `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` variables. You can manually set these variables using the directives below:

``` bash
export NCCL_SOCKET_IFNAME=<network_interface>
export GLOO_SOCKET_IFNAME=<network_interface>
```

</details>

<!-- cell:30 -->

**重要**：对于多节点训练，请确保满足以下要求：

* 步骤5必须在参与训练的**所有节点**上执行。  
* 环境变量 `NCCL_SOCKET_IFNAME` 和 `GLOO_SOCKET_IFNAME` 必须在所有节点上设置为**相同的值**，以确保网络通信一致。

<details markdown="1">
<summary>查看英文原文 · 单元 30</summary>

**Important**: For multi-node training, ensure the following requirements are met:

* Step 5 must be executed on **all nodes** participating in the training.  
* The environment variables `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` must be set to the **same value** across all nodes to ensure consistent network communication.

</details>

<!-- cell:31 -->

<a id="training-configuration"></a>

## 训练配置

在启动训练任务之前，正确配置训练环境至关重要。本节介绍运行 预训练 的基本配置，包括：

* 训练模式（单节点或多节点训练）
* 数据集选项
* 分词器选择

<details markdown="1">
<summary>查看英文原文 · 单元 31</summary>

## Training configuration

Before launching a training task, it's crucial to configure the training environment properly. This section covers the essential configurations for running pretraining, including:

* The training mode (single or multi-node training)
* Dataset options
* Tokenizer selection

</details>

<!-- cell:32 -->

<a id="training-modes"></a>

### 训练模式

训练模式有单节点训练和多节点训练两种。

<a id="single-node-training"></a>

#### 单节点训练

在单节点训练中，所有计算都发生在单个服务器或具有多个 GPU 的机器上。这更容易设置并且足以进行较小规模的实验。

要在单个节点上启动训练，请使用以下命令：

<details markdown="1">
<summary>查看英文原文 · 单元 32</summary>

### Training modes

The two training modes are single-node training and multi-node training.

#### Single-node training

In single-node training, all computations occur on a single server or machine with multiple GPUs. This is simpler to set up and sufficient for smaller-scale experiments.

To launch training on a single node, use the following command:

</details>

```python
!TEE_OUTPUT=1 MBS=2 BS=64 TP=8 TE_FP8=0 SEQ_LENGTH=4096 bash examples/llama/train_llama2.sh
```

<!-- cell:34 -->

上一个命令中的变量是：

* `TEE_OUTPUT=1`：启用到所有节点的输出流。
* `MBS=2`：设置微型批大小。
* `BS=64`：设置全局批大小。
* `TP=8`：配置 张量并行 8 路并行。
* `TE_FP8=0`：禁用 FP8 优化（将其设置为 `1` 即可启用）。
* `SEQ_LENGTH=4096`：指定训练的最大序列长度。

<details markdown="1">
<summary>查看英文原文 · 单元 34</summary>

The variables in the previous command are:

* `TEE_OUTPUT=1`: Enables output streaming to all nodes.
* `MBS=2`: Sets the micro-batch size.
* `BS=64`: Sets the global batch size.
* `TP=8`: Configures tensor parallelism with 8-way parallelism.
* `TE_FP8=0`: Disables FP8 optimizations (set this to `1` to enable).
* `SEQ_LENGTH=4096`: Specifies the maximum sequence length for training.

</details>

<!-- cell:35 -->

<a id="multi-node-training"></a>

#### 多节点训练

对于更大规模的训练，您可以将工作负载分布在多个节点上。多节点训练需要额外的配置才能实现节点之间的通信。

在运行训练脚本之前，请更新以下环境变量：

**主节点地址**：指定主节点的主机名或IP地址。

```bash
MASTER_ADDR="${MASTER_ADDR:-localhost}"
```

**节点数**：定义节点总数。

``` bash
NNODES="${NNODES:-1}"
```

**节点等级**：为每个节点分配唯一的等级，如下所示：

* `0`为主节点。
* `1`、`2`等，用于工作节点。

``` bash
NODE_RANK="${NODE_RANK:-0}"
```

<a id="run-the-training-script"></a>

### 运行训练脚本

使用以下命令在所有节点上执行训练脚本：

<details markdown="1">
<summary>查看英文原文 · 单元 35</summary>

#### Multi-node training

For larger-scale training, you can distribute the workload across multiple nodes. Multi-node training requires additional configuration to enable communication between nodes.

Before running the training script, update the following environment variables:

**Master node address**: Specify the hostname or IP address of the primary node.

```bash
MASTER_ADDR="${MASTER_ADDR:-localhost}"
```

**Number of nodes**:  Define the total number of nodes.

``` bash
NNODES="${NNODES:-1}"
```

**Node rank**: Assign a unique rank to each node, as follows:

* `0` for the primary node.
* `1`, `2`, etc., for the worker nodes.

``` bash
NODE_RANK="${NODE_RANK:-0}"
```

### Run the training script

Execute the training script on all nodes using the following command:

</details>

```python
!TEE_OUTPUT=1 MBS=2 BS=64 TP=8 TE_FP8=0 SEQ_LENGTH=4096 bash examples/llama/train_llama2.sh
```

<!-- cell:37 -->

**重要提示：** 对于多节点训练，请确保在所有节点上运行前面的所有步骤，并按如上所述修改每个节点上的 `NODE_RANK`。

**提示**：在启动全面训练之前，通过模拟训练任务测试多节点通信。这有助于调试任何节点通信或数据集路径问题。

<details markdown="1">
<summary>查看英文原文 · 单元 37</summary>

**Important:** For multi-node training, ensure you run all the previous steps on all the nodes and modify the `NODE_RANK` on each node as described above.

**Tip**: Test multi-node communication with a mock training task before launching full-scale training. This helps debug any node communication or dataset path issues.

</details>

<!-- cell:38 -->

<a id="dataset-options"></a>

### 数据集选项

该数据集是 预训练 的关键组成部分。您可以根据您的要求使用真实数据或模拟数据。

<a id="using-real-data"></a>

#### 使用真实数据

要使用真实数据集：

1. 更新 `DATA_PATH` 变量以指向数据集的位置。

   ``` bash
   DATA_DIR="/root/.cache/data"  # Directory where your dataset is stored
   DATA_PATH=${DATA_DIR}/bookcorpus_text_sentence
   ```

2. 将此变量传递给训练脚本：

   ``` bash
   --data-path $DATA_PATH
   ```

   **注意**：确保数据集文件可在 Docker 容器内访问。

<a id="using-mock-data"></a>

#### 使用模拟数据

要使用模拟数据，请传递此变量：

``` bash
--mock-data
```

<details markdown="1">
<summary>查看英文原文 · 单元 38</summary>

### Dataset options

The dataset is a critical component of pretraining. You can use either real data or mock data based on your requirements.

#### Using real data

To use a real dataset:

1. Update the `DATA_PATH` variable to point to the location of your dataset.

   ``` bash
   DATA_DIR="/root/.cache/data"  # Directory where your dataset is stored
   DATA_PATH=${DATA_DIR}/bookcorpus_text_sentence
   ```

2. Pass this variable to the training script:

   ``` bash
   --data-path $DATA_PATH
   ```

   **Note**: Ensure the dataset files are accessible inside the Docker container.

#### Using mock data

To use mock data, pass this variable:

``` bash
--mock-data
```

</details>

<!-- cell:39 -->

<a id="tokenizer-selection"></a>

### 分词器选择

标记化是将原始文本转换为模型可以处理的标记的过程。不同的 Llama 模型需要特定的标记器：

<a id="for-llama-2-models"></a>

#### 对于 Llama-2 型号：

使用 `Llama2Tokenizer`。

<a id="for-llama-3-and-llama-31-models"></a>

#### 对于 Llama-3 和 Llama-3.1 型号：

使用 `HuggingFaceTokenizer`。在 `TOKENIZER_MODEL` 变量中设置 Hugging Face 模型链接。例如：

``` bash
TOKENIZER_MODEL=meta-llama/Llama-3.1-8B
```

<details markdown="1">
<summary>查看英文原文 · 单元 39</summary>

### Tokenizer selection

Tokenization is the process of converting raw text into tokens that the model can process. Different Llama models require specific tokenizers:

#### For Llama-2 models:

Use the `Llama2Tokenizer`.

#### For Llama-3 and Llama-3.1 models:

Use the `HuggingFaceTokenizer`. Set the Hugging Face model link in the `TOKENIZER_MODEL` variable. For example:

``` bash
TOKENIZER_MODEL=meta-llama/Llama-3.1-8B
```

</details>

<!-- cell:40 -->

<a id="next-steps"></a>

## 后续步骤

继续阅读[用威震天-LM 训练 Llama-3.1 8B](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/train_llama_mock_data.html) 指南，您将：

* 使用本教程中设置的环境和配置。
* 使用模拟数据运行实际的 预训练 示例。

<details markdown="1">
<summary>查看英文原文 · 单元 40</summary>

## Next steps

Proceed to the [Training Llama-3.1 8B with Megatron-LM](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/train_llama_mock_data.html) guide, where you will:

* Use the environment and configurations set up in this tutorial.
* Run practical pretraining examples using mock data.

</details>
