> 原文作者：Ning Zhang  
> [AMD 官方教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/triton_kernel_dev.html) · [固定版本源码](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/gpu_dev_optimize/triton_kernel_dev.ipynb)  
> 上游版本：`0071bfd4bd3bd9525405c09a6ae69c983b09312a` · 同步日期：2026-10-12  
> 翻译状态：机器初译，待逐篇人工校对；代码与命令保持原文，未在 AMD GPU 上运行验证。

<!-- cell:1 -->

<a id="kernel-development-and-optimization-with-triton"></a>

# 使用 Triton 开发与优化内核
**作者**：Ning Zhang
**知识水平**：中级

[![在 AMD Developer Cloud 中打开](../../original/assets/jupyter_notebook_button_fnl_small.png)](https://notebooks.amd.com/tutorials/triton_kernel_dev.ipynb)

[OpenAI Triton](https://github.com/triton-lang/triton) 是一种受 AMD GPU 支持的开源编程语言，旨在简化高性能任务的 GPU 编程，特别是在 AI 应用程序中。本教程演示如何在 AMD GPU 上设置 Triton 开发环境并优化 Triton 内核性能。 

<a id="prerequisites"></a>

## 前置条件

本教程是使用以下设置进行开发和测试的。 

<a id="operating-system"></a>

### 操作系统

* **Ubuntu 22.04/24.04**：确保您的系统运行的是 Ubuntu 版本 22.04 或 24.04。

<a id="hardware"></a>

### 硬件

* **AMD Instinct™ GPU**：本教程在 AMD Instinct MI300X GPU 上进行了测试。确保您使用的是 AMD Instinct GPU 或支持 ROCm 的兼容硬件，并且您的系统满足[官方要求](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)。


<a id="software"></a>

### 软件

* **ROCm 6.2、6.3 或 6.4**：按照 [ROCm 安装指南](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html) 安装并验证 ROCm。 

  安装后，使用 `amd-smi` 命令确认您的设置。 AMD 还提供预构建的 ROCm Docker 映像，例如 [ROCm PyTorch 映像](https://hub.docker.com/r/rocm/pytorch)、[ROCm Ubuntu 22.04 映像](https://hub.docker.com/r/rocm/dev-ubuntu-22.04) 和 [ROCm Ubuntu 24.04 映像](https://hub.docker.com/r/rocm/dev-ubuntu-24.04)。您可以使用这些预构建的 Docker 映像来减少设置 ROCm 环境所需的工作量。
  
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

<a id="set-up-the-triton-development-environment"></a>

## 设置Triton开发环境 

本教程使用预构建的 ROCm PyTorch 映像，但您也可以尝试其他 ROCm 环境作为基础映像。 

<a id="step-1-launch-the-docker-image"></a>

### 第 1 步：启动 Docker 镜像

启动 Docker 容器。将 `/path/to/Triton_Sample` 替换为主机上 Triton 示例代码所在目录的完整路径。

```bash
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
  rocm/pytorch:latest
```

**注意**：该命令将当前目录挂载到容器中的`/workspace`目录中。确保在运行 Docker 命令之前将Notebook文件复制到此目录，或者在启动后上传到 Jupyter Notebook 环境。保存终端输出中提供的令牌或 URL，以便从 Web 浏览器访问Notebook。您可以从 [AI Developer Hub GitHub 存储库](https://github.com/ROCm/gpuaidev) 下载此Notebook。

<a id="step-2-launch-jupyter-notebooks-in-the-container"></a>

### 步骤 2：在容器中启动 Jupyter Notebooks

在 Docker 容器内，使用以下命令安装 Jupyter：

``` bash
pip install jupyter
```

启动 Jupyter 服务器：

``` bash
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**注意**：在运行上述命令之前，请确保您的系统上尚未使用端口 `8888`。如果是，您可以通过将 `--port=8888` 替换为其他端口号来指定其他端口，例如 `--port=8890`。
**注意**：将本教程上传到服务器后，本教程的其余部分可以在 Jupyter Notebook 中作为交互式块运行。

<a id="step-3-install-openai-triton"></a>

### 步骤 3：安装 OpenAI Triton

在安装正确版本的 OpenAI Triton 之前，您必须卸载所有旧版本。

<a id="1-uninstall-the-old-version-of-triton"></a>

#### 1.卸载旧版本的Triton

强烈建议您在项目中使用最新版本的 Triton。 AMD 和其他供应商经常在 [OpenAI Triton](https://github.com/triton-lang/triton) 中更新他们的优化过程和算法。这些更新可以提高您的 Triton 内核性能。

<details markdown="1">
<summary>查看英文原文 · 单元 1</summary>

# Kernel development and optimization with Triton

**Author**: Ning Zhang  
**Knowledge level**: Intermediate

[![Open in AMD Developer Cloud](../../original/assets/jupyter_notebook_button_fnl_small.png)](https://notebooks.amd.com/tutorials/triton_kernel_dev.ipynb)

[OpenAI Triton](https://github.com/triton-lang/triton) is an open-source programming language that is supported by AMD GPUs and is designed to simplify GPU programming for high-performance tasks, particularly in AI applications. This tutorial demonstrates how to set up the Triton development environment and optimize Triton kernel performance on AMD GPUs. 

## Prerequisites

This tutorial was developed and tested using the following setup. 

### Operating system

* **Ubuntu 22.04/24.04**: Ensure your system is running Ubuntu version 22.04 or 24.04.

### Hardware

* **AMD Instinct™ GPUs**: This tutorial was tested on an AMD Instinct MI300X GPU. Ensure you are using an AMD Instinct GPU or compatible hardware with ROCm support and that your system meets the [official requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).


### Software

* **ROCm 6.2, 6.3, or 6.4**: Install and verify ROCm by following the [ROCm install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html). 

  After installation, confirm your setup using the `amd-smi` command. AMD also provides prebuilt ROCm Docker images, for example, a [ROCm PyTorch image](https://hub.docker.com/r/rocm/pytorch), [ROCm Ubuntu 22.04 image](https://hub.docker.com/r/rocm/dev-ubuntu-22.04), and [ROCm Ubuntu 24.04 image](https://hub.docker.com/r/rocm/dev-ubuntu-24.04). You can use these prebuilt Docker images to reduce the effort required to set up a ROCm environment.
  
  **Note**: For ROCm 6.4 and earlier, use the `rocm-smi` command instead of `amd-smi`.

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

## Set up the Triton development environment 

This tutorial uses the prebuilt ROCm PyTorch image, but you can also try other ROCm environments as the base image. 

### Step 1: Launch the Docker image

Launch the Docker container. Replace `/path/to/Triton_Sample` with the full path to the directory on your host machine where the Triton sample code is located.

```bash
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
  rocm/pytorch:latest
```

**Note**: This command mounts the current directory to the `/workspace` directory in the container. Ensure the notebook file is either copied to this directory before running the Docker command or uploaded into the Jupyter Notebook environment after it starts. Save the token or URL provided in the terminal output to access the notebook from your web browser. You can download this notebook from the [AI Developer Hub GitHub repository](https://github.com/ROCm/gpuaidev).

### Step 2: Launch Jupyter Notebooks in the container

Inside the Docker container, install Jupyter using the following command:

``` bash
pip install jupyter
```

Start the Jupyter server:

``` bash
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**Note**: Ensure port `8888` is not already in use on your system before running the above command. If it is, you can specify a different port by replacing `--port=8888` with another port number, for example, `--port=8890`.

**Note**: The rest of this tutorial can run as interactive blocks in your Jupyter notebook after you upload this tutorial to your server.

### Step 3: Install OpenAI Triton

Before you can install the correct version of OpenAI Triton, you must uninstall any old versions.

#### 1. Uninstall the old version of Triton

It's strongly recommended that you use the latest version of Triton in your project. AMD and other vendors frequently update their optimization passes and algorithms in [OpenAI Triton](https://github.com/triton-lang/triton). These updates can improve your Triton kernel performance.

</details>

```python
!pip uninstall -y triton
```

<!-- cell:3 -->

<a id="2-install-openai-triton-from-the-source-code"></a>

#### 2. 从源代码安装 OpenAI Triton

下面列出了安装 Triton 的详细步骤。

**注意：** 如果您在构建 Triton 时有任何疑问或问题，请将其提交至 [Triton Issues](https://github.com/triton-lang/triton/issues)。

<details markdown="1">
<summary>查看英文原文 · 单元 3</summary>

#### 2. Install OpenAI Triton from the source code

The detailed steps to install Triton are listed below.

**Note:** If you have any questions or issues when building Triton, submit them to [Triton Issues](https://github.com/triton-lang/triton/issues).

</details>

```python
%%bash
# Remove existing Triton folder if it exists
if [ -d "triton" ]; then
    echo "Removing existing triton directory..."
    rm -rf triton
fi

# Clone Triton repo
git clone https://github.com/triton-lang/triton.git

# Install dependencies and Triton from source (non-editable install)
cd triton
pip install -r python/requirements.txt
pip install .
```

<!-- cell:5 -->

<a id="step-4-validate-triton-on-an-amd-gpu"></a>

### 步骤 4：在 AMD GPU 上验证 Triton

Triton安装成功后，验证其在AMD GPU上是否正常工作。在 Python 中运行以下向量相加示例，以确认 Triton 内核提供与 Torch API 类似的结果，这意味着它在 AMD GPU 上高效执行。

<details markdown="1">
<summary>查看英文原文 · 单元 5</summary>

### Step 4: Validate Triton on an AMD GPU

After Triton is successfully installed, validate whether it works properly on an AMD GPU. Run the following vector-add sample in Python to confirm that the Triton kernel provides similar results as the Torch APIs, which means it's performing efficiently on AMD GPUs.

</details>

```python
import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)


# %%
# Let's also declare a helper function to (1) allocate the `z` tensor
# and (2) enqueue the above kernel with appropriate grid/block sizes:

def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output


# %%
# We can now use the above function to compute the element-wise sum of two `torch.tensor` objects and test its correctness:

torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
```

<!-- cell:7 -->

输出日志为：

    张量([1.3713, 1.3076, 0.4940, ..., 0.6724, 1.2141, 0.9733], 设备='cuda:0')
    张量([1.3713, 1.3076, 0.4940, ..., 0.6724, 1.2141, 0.9733], 设备='cuda:0')
    torch 和 triton 之间的最大差异是 0.0

<details markdown="1">
<summary>查看英文原文 · 单元 7</summary>

The output log is:

    tensor([1.3713, 1.3076, 0.4940,  ..., 0.6724, 1.2141, 0.9733], device='cuda:0')
    tensor([1.3713, 1.3076, 0.4940,  ..., 0.6724, 1.2141, 0.9733], device='cuda:0')
    The maximum difference between torch and triton is 0.0

</details>

<!-- cell:8 -->

<a id="optimize-the-triton-code-for-amd-gpus"></a>

## 针对 AMD GPU 优化 Triton 代码

softmax 函数常用于卷积神经网络 (CNN) 分类模型，甚至基于 Transformer 的 LLM 模型。它通过获取每个值的指数并将这些值除以所有指数的总和来标准化这些值，从而将原始输出分数或 logits 转换为概率。此过程确保输出值在 (0,1) 范围内且总和为 1，以便将它们解释为概率。 PyTorch 将此函数实现为[标准 API](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html)。 

<a id="naive-version"></a>

### 幼稚版本 

根据规范，您在 Triton 内核中实现了 softmax 算法的简单版本。为了确定最大数据点和所有指数的相应总和，该内核版本使用两个 for 循环，再加上一个 for 循环来计算最终的 softmax 结果，总共三个循环。 

以下示例在 8192 x 8192 张量上测试内核性能​​，列维度的块大小为 256。运行预热部分以避免在最终数据中包含内核编译时间后，您可以获得 naive 版本的性能数据。

<details markdown="1">
<summary>查看英文原文 · 单元 8</summary>

## Optimize the Triton code for AMD GPUs

The softmax function is often used in convolutional neural network (CNN) classification models and even Transformer-based LLM models. It converts raw output scores, or logits, into probabilities by taking the exponential of each value and normalizing these values by dividing by the sum of all the exponentials. This process ensures that the output values are in the range (0,1) and sum to 1 to allow them to be interpreted as probabilities. PyTorch implements this function as [a standard API](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html). 

### Naive version 

According to the specification, you implemented the naive version of the softmax algorithm in the Triton kernel. To determine the maximum data point and the corresponding sum of all the exponentials, this kernel version uses two for-loops, along with one more for-loop to calculate the final softmax result, for a total of three loops in all. 

The following example tests the kernel performance on an 8192 x 8192 tensor, with a block size for the column dimension of 256. After running the warmup section to avoid including the kernel compilation time in the final data, you can obtain the performance data for the naive version.

</details>

```python
import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def softmax_kernel_naive(in_ptr, output_ptr, row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    in_max = -float('inf')
    for offset in range(0, n_cols, BLOCK_SIZE):
        col_range = tl.arange(0, BLOCK_SIZE)
        col_mask = col_range + offset < n_cols
        in_data = tl.load(in_ptr + pid * row_stride + col_range + offset, mask=col_mask, other=-float('inf'))
        in_max = tl.maximum(in_max, tl.max(in_data, axis=-1))
    
    in_exp_sum = 0.0
    for offset in range(0, n_cols, BLOCK_SIZE):
        col_range = tl.arange(0, BLOCK_SIZE)
        col_mask = col_range + offset < n_cols
        in_data = tl.load(in_ptr + pid * row_stride + col_range + offset, mask=col_mask, other=-float('inf'))
        in_exp_sum = in_exp_sum + tl.sum(tl.exp(in_data - in_max), axis=-1)
    
    for offset in range(0, n_cols, BLOCK_SIZE):
        col_range = tl.arange(0, BLOCK_SIZE)
        col_mask = col_range + offset < n_cols
        in_data = tl.load(in_ptr + pid * row_stride + col_range + offset, mask=col_mask)
        in_exp = tl.exp(in_data - in_max)
        tl.store(output_ptr + pid * row_stride + col_range + offset, in_exp / in_exp_sum, mask=col_mask)

torch.manual_seed(0)
x = torch.randn(8192, 8192, device=DEVICE)
n_rows, n_cols = x.shape
output_triton = torch.empty_like(x)
BLOCK_SIZE = 256
temp = torch.randn(n_rows, n_cols, device=DEVICE)
softmax_kernel_naive[(n_rows,)](
        temp,
        output_triton,
        temp.stride(0),
        n_cols,
        BLOCK_SIZE
)#warmup
torch.cuda.empty_cache() #clean cache

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
softmax_kernel_naive[(n_rows,)](
        x,
        output_triton,
        x.stride(0),
        n_cols,
        BLOCK_SIZE
)
end_event.record()

torch.cuda.synchronize()
elapsed_time_ms = start_event.elapsed_time(end_event)
print(f'Softmax Triton Naive Version Elapsed: {elapsed_time_ms:.3f}ms')
```

<!-- cell:10 -->

<a id="online-softmax-version"></a>

### 在线softmax版本 

使用Triton语言实现算法很容易。为了从当前内核获得更好的性能，首先确定是否有更有效的算法或解决方案。如果是这样，请在 Triton 内核中尝试新算法。为了减少朴素softmax算法中三个for循环引起的内存访问，[Online Normalizer Calculator for softmax](https://arxiv.org/pdf/1805.02867)论文中提出了一种新的在线softmax算法。

下面的代码按照网上的softmax算法，对naive版本的kernel做了一些修改。

<details markdown="1">
<summary>查看英文原文 · 单元 10</summary>

### Online softmax version 

It's easy to use the Triton language to implement algorithms. To obtain better performance from the current kernel, first determine whether there's a more efficient algorithm or solution. If so, try the new algorithm in your Triton Kernel. To reduce the memory accesses caused by the three for-loops in the naive softmax algorithm, a new online softmax algorithm has been proposed in the [Online normalizer calculation for softmax](https://arxiv.org/pdf/1805.02867) paper.

In accordance with the online softmax algorithm, the following code makes a few modifications to the naive version kernel.

</details>

```python
import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def softmax_kernel_v1(in_ptr, output_ptr, row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    in_max = -float('inf')
    in_exp_sum = 0.0
    for offset in range(0, n_cols, BLOCK_SIZE):
        col_range = tl.arange(0, BLOCK_SIZE)
        col_mask = col_range + offset < n_cols
        in_data = tl.load(in_ptr + pid * row_stride + col_range + offset, mask=col_mask, other=-float('inf'))
        in_max_new = tl.maximum(in_max, tl.max(in_data, axis=-1))
        in_exp_sum = in_exp_sum * tl.exp(in_max - in_max_new) + tl.sum(tl.exp(in_data - in_max_new), axis=-1)
        in_max = in_max_new
    
    for offset in range(0, n_cols, BLOCK_SIZE):
        col_range = tl.arange(0, BLOCK_SIZE)
        col_mask = col_range + offset < n_cols
        in_data = tl.load(in_ptr + pid * row_stride + col_range + offset, mask=col_mask)
        in_exp = tl.exp(in_data - in_max)
        tl.store(output_ptr + pid * row_stride + col_range + offset, in_exp / in_exp_sum, mask=col_mask)

torch.manual_seed(0)
x = torch.randn(8192, 8192, device=DEVICE)
n_rows, n_cols = x.shape
output = torch.empty_like(x)
BLOCK_SIZE = 256
temp = torch.randn(n_rows, n_cols, device=DEVICE)
softmax_kernel_v1[(n_rows,)](
        temp,
        output,
        temp.stride(0),
        n_cols,
        BLOCK_SIZE
)#warmup
torch.cuda.empty_cache() #clean cache

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
softmax_kernel_v1[(n_rows,)](
        x,
        output,
        x.stride(0),
        n_cols,
        BLOCK_SIZE
)
end_event.record()

torch.cuda.synchronize()
elapsed_time_ms = start_event.elapsed_time(end_event)

print(f'Softmax Triton V1 Version Elapsed: {elapsed_time_ms:.3f}ms')
```

<!-- cell:12 -->

<a id="fused-softmax-version"></a>

### Fused-softmax 版本

OpenAI Triton 提供了“fused-softmax”softmax 参考示例。在在线softmax算法的基础上，继续简化最大数据计算，去掉1个for循环。它还告诉编译器通过增加扭曲数量来每行使用更多线程。通常会调整此配置以获得更好的性能。最后，通过调整GPU硬件配置，改进了内核启动方案，可以带来更高的GPU 内核占用率和更好的性能。

<details markdown="1">
<summary>查看英文原文 · 单元 12</summary>

### Fused-softmax version

OpenAI Triton provides the "fused-softmax" softmax reference example. Based on the online softmax algorithm, it continues to simplify the maximum data calculation by removing one for-loop. It also tells the compiler to use more threads per row by increasing the number of warps. This configuration is often tuned for better performance. Finally, it improves the kernel launching scheme by adjusting the GPU hardware configuration, which can lead to higher GPU kernel occupancy and better performance.

</details>

```python
import torch
import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942',
                                                                                   'gfx90a', 'gfx908')

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}

torch.manual_seed(0)
x = torch.randn(8192, 8192, device=DEVICE)
n_rows, n_cols = x.shape
# Allocate output
y = torch.empty_like(x)

# The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
BLOCK_SIZE = triton.next_power_of_2(n_cols)

# Another trick we can use is to ask the compiler to use more threads per row by
# increasing the number of warps (`num_warps`) over which each row is distributed.
num_warps = 8

# Number of software pipelining stages.
num_stages = 4 if SIZE_SMEM > 200000 else 2

# pre-compile kernel to get register usage and compute thread occupancy.
kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                   num_stages=num_stages, num_warps=num_warps, grid=(1, ))
kernel._init_handles()
n_regs = kernel.n_regs
size_smem = kernel.metadata.shared

if is_hip():
    # NUM_REGS represents the number of regular purpose registers. On CDNA architectures this is half of all registers available.
    # However, this is not always the case. In most cases all registers can be used as regular purpose registers.
    # ISA SECTION (3.6.4 for CDNA3)
    # VGPRs are allocated out of two pools: regular VGPRs and accumulation VGPRs. Accumulation VGPRs are used
    # with matrix VALU instructions, and can also be loaded directly from memory. A wave may have up to 512 total
    # VGPRs, 256 of each type. When a wave has fewer than 512 total VGPRs, the number of each type is flexible - it is
    # not required to be equal numbers of both types.
    if is_cdna():
        NUM_GPRS = NUM_REGS * 2

    # MAX_NUM_THREADS represents maximum number of resident threads per multi-processor.
    # When we divide this number with WARP_SIZE we get maximum number of waves that can
    # execute on a CU (multi-processor)  in parallel.
    MAX_NUM_THREADS = properties["max_threads_per_sm"]
    max_num_waves = MAX_NUM_THREADS // WARP_SIZE
    occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps
else:
    occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    
occupancy = min(occupancy, SIZE_SMEM // size_smem)
num_programs = NUM_SM * occupancy

num_programs = min(num_programs, n_rows)

# Create a number of persistent programs.
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages)
end_event.record()

torch.cuda.synchronize()
elapsed_time_ms = start_event.elapsed_time(end_event)
print(f'Softmax Triton V2 Version Elapsed: {elapsed_time_ms:.3f}ms')
```

<!-- cell:14 -->

<a id="summary"></a>

## 总结 
在本教程中，您学习了如何在 AMD GPU 上开发和优化 Triton 内核。要了解有关 OpenAI Triton 的更多信息，请参阅[Triton 官方文档](https://triton-lang.org/main/index.html)。要了解有关在 AMD GPU 上运行 Triton 的更多信息，请参阅 [ROCm Triton 优化](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-triton-kernel.html) 和 [Triton 博客上的内核开发优化](https://rocm.blogs.amd.com/software-tools-optimization/kernel-development-optimizations-with-triton-on-/README.html)。希望本教程能够鼓励您在 AMD GPU 上对 Triton 进行调整、测试和贡献，并帮助塑造 AI 加速的未来。

<details markdown="1">
<summary>查看英文原文 · 单元 14</summary>

## Summary 
In this tutorial, you learned how to develop and optimize Triton kernels on AMD GPUs. To learn more about OpenAI Triton, see the [official Triton documentation](https://triton-lang.org/main/index.html). To find out more about running Triton on AMD GPUs, see [ROCm Triton optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-triton-kernel.html) and the [Kernel development optimization on Triton blog](https://rocm.blogs.amd.com/software-tools-optimization/kernel-development-optimizations-with-triton-on-/README.html). Hopefully, this tutorial encourages you to tune, test, and contribute to Triton on AMD GPUs and help shape the future of AI acceleration.

</details>
