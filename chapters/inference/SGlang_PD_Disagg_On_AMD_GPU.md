> 原文作者：Ning Zhang  
> [AMD 官方教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/SGlang_PD_Disagg_On_AMD_GPU.html) · [固定版本源码](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/inference/SGlang_PD_Disagg_On_AMD_GPU.ipynb)  
> 上游版本：`0071bfd4bd3bd9525405c09a6ae69c983b09312a` · 同步日期：2026-10-12  
> 翻译状态：机器初译，待逐篇人工校对；代码与命令保持原文，未在 AMD GPU 上运行验证。

[中文 Notebook](../../notebooks/inference/SGlang_PD_Disagg_On_AMD_GPU.ipynb) · [英文原文](../../original/inference/SGlang_PD_Disagg_On_AMD_GPU.md) · [中英对照](../../bilingual/inference/SGlang_PD_Disagg_On_AMD_GPU.md)

---

<a id="llm-distributed-inference-and-pd-disaggregation-on-amd-instinct-gpus"></a>

# 使用 SGLang 分离预填充与解码
**作者**：Ning Zhang
**知识水平**：中级

随着LLM模型规模的快速增长，单节点推理优化开始显示出其在LLM服务扩展方面的局限性。多个节点上的分布式 推理 对于高效的 LLM 服务变得更加重要。 预填充 和解码 (PD) 分解是 GPU 节点上的 LLM 分布式 推理 的典型用例。 LLM 推理 包含两个不同的阶段：预填充 和解码。 预填充 阶段是计算密集型，处理整个输入序列，而解码阶段是内存密集型，管理用于令牌生成的键值 (KV) 缓存。 预填充与解码分离在不同的GPU节点上独立运行这两个阶段，这提供了高效的GPU资源分配和独立的性能调优的好处。本教程演示如何在一个或两个具有 AMD Instinct™ GPU 的节点上设置 1P1D 分布式 推理。 

<a id="prerequisites"></a>

## 前置条件

本教程是使用以下设置进行开发和测试的。

<a id="operating-system"></a>

### 操作系统

* **Ubuntu 22.04/24.04**：确保您的系统运行的是 Ubuntu 22.04 或 24.04。

<a id="hardware"></a>

### 硬件

- **AMD Instinct GPU**：在 MI300X 上测试。适用于单个 MI300X 节点或两个 MI300X 节点（每个节点有 8 个 MI300X GPU）。
- **RDMA NIC**（单节点和两节点需要通过 RDMA 传输 KV 缓存）：
  - 使用支持 RDMA 的 NIC（例如 Broadcom Thor2/BCM-57608 或 NVIDIA/Mellanox ConnectX）。
  - 安装适当的供应商驱动程序和 RDMA 用户空间 (`rdma-core/libibverbs`)。使用以下命令进行验证：
    - `ibv_devices` 和 `ibv_devinfo`
    - `ls /dev/infiniband`
  - 如果在 Docker 中运行，请确保 `/dev/infiniband` 映射到容器中（请参阅下面的启动命令）。
  - 对于双节点配置，RDMA 必须在两个节点（具有 PFC 或 InfiniBand 的 RoCEv2）上均可用，并且必须正确布线并配置交换机。
- **ROCm 兼容性**：使用支持 ROCm 的 AMD Instinct GPU，并确保您的系统满足 [官方要求](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)。
<a id="software"></a>

### 软件

- **ROCm 6.3 或更高版本**（主机）：按照 [ROCm 安装指南](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html) 安装并验证 ROCm。安装后，使用 `amd-smi` 命令确认您的设置。

  **注意**：对于 ROCm 6.4 及更早版本，请改用 `rocm-smi` 命令。
- **Docker**（主机）：在 Linux 上安装 Docker 引擎并使用 `docker --version` 命令进行验证。您将在容器内运行本教程。确保主机允许您映射 GPU 和 RDMA 设备。
  - 启动容器时，映射设备并使用推荐的标志：
    - 设备：`/dev/kfd`、`/dev/dri`、`/dev/infiniband` 和 `/dev/infiniband/rdma_cm`
    - 标志：`--network=host --ipc=host --shm-size 32G --group-add=video`
- **预构建 ROCm Docker 镜像**（建议减少设置工作量）：
  - [SGLang ROCm 图像](https://hub.docker.com/r/lmsysorg/sglang/tags)
  - [ROCm Ubuntu 22.04](https://hub.docker.com/r/rocm/dev-ubuntu-22.04)
  - [ROCm Ubuntu 24.04](https://hub.docker.com/r/rocm/dev-ubuntu-24.04)
- **Jupyter（在容器中）**：使用 `pip install jupyter` 安装以运行此Notebook。

<a id="hugging-face-api-access"></a>

### Hugging Face API 访问
* 从[Hugging Face](https://huggingface.co)获取API令牌用于下载模型。
* 确保 Hugging Face API 令牌具有访问所需检查点所需的权限和批准：
  - 对于一个节点，您必须有权访问 [Meta Llama Llama-3.1-8B-Instruct 检查点](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)。
  - 对于两个节点，您必须有权访问 [Meta Llama Llama-3.3-70B-Instruct 检查点](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)。

<a id="set-up-the-pd-disaggregation-environment"></a>

## 设置预填充与解码分离环境 

在本教程中，您将使用预构建的 ROCm SGLang 映像，该映像将 SGLang 与 AMD [ROCm](https://rocm.docs.amd.com/en/latest/index.html) 软件堆栈集成。如果需要，您还可以尝试其他 ROCm 映像作为基础映像。  

<a id="step-1-prepare-the-tutorial-environment"></a>

### 第1步：准备教程环境

请按照以下步骤配置您的教程环境：

<a id="pull-the-docker-image"></a>

#### 拉取 Docker 镜像

使用`lmsysorg/sglang:v0.4.9-rocm630` Docker镜像作为基础镜像。这是为本教程测试的最新图像。

**注意**：SGLang 社区继续发布其他 ROCm SGLang Docker 映像。强烈建议您尝试最新的可用映像以获得更好的性能。

``` bash
docker pull lmsysorg/sglang:v0.4.9-rocm630
```

<a id="launch-the-docker-container"></a>

#### 启动 Docker 容器

为了获得良好的网络传输性能，运行 预填充与解码分离 的节点需要 RDMA NIC。启动 Docker 映像时，将 RDMA 设备映射到 Docker 容器，如下面的命令所示。

``` bash
docker run -it --rm \
  --network=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  --ipc=host \
  --device=/dev/infiniband \
  --device=/dev/infiniband/rdma_cm \
  --privileged \
  --cap-add=SYS_ADMIN \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size 32G \
  -v $(pwd):/workspace \
  -w /workspace \
  lmsysorg/sglang:v0.4.9-rocm630
```

**注意**：该命令将当前目录挂载到容器中的`/workspace`目录中。确保在运行 Docker 命令之前将Notebook文件复制到此目录，或者在启动后上传到 Jupyter Notebook 环境。保存终端输出中提供的令牌或 URL，以便从 Web 浏览器访问Notebook。您可以从 [AI Developer Hub GitHub 存储库](https://github.com/ROCm/gpuaidev) 下载此Notebook。

<a id="install-and-launch-jupyter"></a>

#### 安装并启动 Jupyter

在 Docker 容器内，使用以下命令安装 Jupyter：

``` bash
pip install jupyter
```

启动 Jupyter 服务器：

``` bash
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**注意**：在运行上述命令之前，请确保您的系统上尚未使用端口 `8888`。如果是，您可以通过将 `--port=8888` 替换为其他端口号来指定其他端口，例如 `--port=8890`。

**注意**：此Notebook的其余部分设计为作为 Jupyter Notebook运行。该Notebook演示了 AMD Instinct GPU 上的 预填充 和解码 (PD) 分解。默认情况下，它在单个节点（节点内 1P1D）上运行。双节点（节点间 1P1D）步骤是可选的并且有明确标记。

**运行模式**

* 单节点（默认）：不需要 [etcd](https://etcd.io/) 键值存储服务器。使用“节点内 1P1D”部分。
* 两个节点（可选）：需要设置 etcd 服务器、RDMA 和 SSH。使用双节点“节点间 1P1D”部分。

<a id="provide-your-hugging-face-token"></a>

#### 提供您的Hugging Face令牌

您需要 Hugging Face API 令牌才能访问具有适当权限的 Llama 模型，如本Notebook前面部分所述。首先，安装 Hugging Face Hub 库。

```python
!pip install --upgrade huggingface_hub
```

在 Jupyter Notebook中运行以下交互式块来设置令牌：

```python
from huggingface_hub import notebook_login, HfApi

# Prompt the user to log in
notebook_login()
```

验证您的令牌是否被正确接受：

```python
# Validate the token
try:
    api = HfApi()
    user_info = api.whoami()
    print(f"Token validated successfully! Logged in as: {user_info['name']}")
except Exception as e:
    print(f"Token validation failed. Error: {e}")
```

<a id="step-2-install-the-necessary-software-components"></a>

### 第 2 步：安装必要的软件组件 

对于节点内1P1D模式，必须安装Mooncake传输引擎。这种情况下不需要 etcd 服务器。对于Inter-node 1P1D模式，您必须同时安装etcd和Mooncake传输引擎。

<a id="step-21-optional-install-etcd"></a>

#### 步骤 2.1：（可选）安装 etcd

如果是单机测试，可以跳过此步骤。 etcd 是一个高度一致的分布式键值存储，它提供了一种可靠的方式来存储需要由分布式系统或机器集群访问的数据。在SGLang 预填充与解码分离解决方案设计中，每个GPU节点上都需要etcd服务器来提供集群元数据存储。必须安装它才能在两个节点上运行本教程。

```python
%%bash
cd /sgl-workspace
apt update && apt install -y wget
wget https://github.com/etcd-io/etcd/releases/download/v3.6.0-rc.5/etcd-v3.6.0-rc.5-linux-amd64.tar.gz -O /tmp/etcd.tar.gz
tar -xvf /tmp/etcd.tar.gz -C /usr/local/bin/ --strip-components=1 && rm /tmp/etcd.tar.gz
```

<a id="step-22-install-mooncake"></a>

#### 步骤2.2：安装Mooncake

Mooncake 采用以 KV 缓存 为中心的分解架构，将 预填充 和解码集群分开。其核心组件，如传输引擎，已集成到SGLang 预填充与解码分离解决方案中，以在节点之间传输KV 缓存。

```python
%%bash
apt update && apt install -y zip unzip openssh-server
apt -y install gcc make libtool autoconf  librdmacm-dev rdmacm-utils infiniband-diags ibverbs-utils perftest ethtool  libibverbs-dev rdma-core strace
cd /sgl-workspace
pip install mooncake-transfer-engine
```

<a id="step-3-install-rdma-related-libraries-and-add-the-network-transfer-configuration"></a>

### 步骤3：安装RDMA相关库并添加网络传输配置

对于单节点1P1D测试，可以跳过步骤3.4中的网络配置。但是，即使对于节点内运行，仍然需要 SGLang/Mooncake 传输引擎来启用 RDMA NIC。使用 `ibv_devices` 验证 RDMA 可用性。如果未列出任何设备，请按照步骤 3.1 中所述安装相应的 RDMA NIC 驱动程序。

<a id="step-31-install-the-nic-rdma-driver"></a>

#### 步骤3.1：安装网卡RDMA驱动程序 

这些指令在配备 Broadcom Thor2/BCM-57608 RDMA NIC（单节点和双节点）的 Instinct MI300X 系统上进行了验证。如果您使用不同的 RDMA NIC，请根据文档下载并安装适当的供应商驱动程序。以下命令仅适用于 Thor2/BCM-57608 NIC。

⚠️ **注意**：特定于驱动程序的说明：以下命令适用于 Instinct MI300X 系统上的 Broadcom Thor2/BCM‑57608 RDMA NIC。如果您的系统使用不同的 RDMA NIC，请按照文档从 NIC 供应商处获取并安装正确的驱动程序和用户空间，然后再继续。您可以通过运行 `lspci | grep -i Eth` 找到您的 NIC 供应商。

```python
%%bash
echo -e "\n\n============Installing required pkgs============\n\n"
apt -y install libelf-dev

cd bcm5760x_230.2.52.0a/drivers_linux/bnxt_rocelib
tar xvf libbnxt_re-230.2.52.0.tar.gz
cd libbnxt_re-230.2.52.0
echo -e "\n\n============Compiling RoCE Lib now============\n\n"
sh autogen.sh
./configure
make
find /usr/lib64/  /usr/lib -name "libbnxt_re-rdmav*.so"  -exec mv {} {}.inbox \;
make install all
sh -c "echo /usr/local/lib >> /etc/ld.so.conf"
ldconfig
cp -f bnxt_re.driver /etc/libibverbs.d/
find . -name "*.so" -exec md5sum {} \;
BUILT_MD5SUM=$(find . -name "libbnxt_re-rdmav*.so" -exec md5sum {} \; |  cut -d " " -f 1)
echo -e "\n\nmd5sum of the built libbnxt_re is $BUILT_MD5SUM"
```

列出 RDMA 设备。

```python
%%bash
ibv_devices
```

运行上述步骤后，使用 `ibv_devices` 命令列出 RDMA 设备，并使用 `ibv_devinfo` 命令检查 RDMA 设备信息。 `ibv_devices` 命令的输出应类似于以下结果，确认可以枚举所有 RDMA 设备并成功运行。

    设备节点 GUID
    ------------------ ----------------
    rdma0 d604e6fffe0e9fb4
    rdma1 d604e6fffe0e9938
    xeth0 d604e6fffee921d0
    rdma2 d604e6fffe780000
    RDMA3 d604e6fffe0e9d34
    rdma4 d604e6fffe780370
    RDMA5 d604e6fffe0e8678
    rdma6 d604e6fffe0e8718
    rdma7 d604e6fffe7801f4

<a id="step-32-build-and-install-the-rocm-aware-ucx-library"></a>

#### 步骤 3.2：构建并安装 ROCm 感知的 UCX 库

[统一通信框架](https://github.com/openucx/ucx) (UCX) 是一个开源的跨平台框架，旨在为各种网络编程模型和接口提供一组通用的通信接口。 UCX 使用 ROCm 技术来实现各种网络操作原语。它是 InfiniBand 和融合以太网 RDMA (RoCE) 网络互连的标准通信库。

```python
%%bash
apt-get install -y flex
git clone https://github.com/openucx/ucx.git -b v1.18.1 
cd ucx 
./autogen.sh
./configure --with-rocm=/opt/rocm --enable-mt --prefix=/opt/ucx 
make -j 
make install
```

为 Jupyter 会话设置 UCX 环境变量，以便可以发现 `ucx_info` 和 UCX 库。这些更改仅影响当前内核。

```python
import os
os.environ['PATH'] = os.environ['PATH'] + ':/opt/ucx/bin'
os.environ['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH'] + ':/opt/ucx/lib'
```

通过运行此命令验证 UCX ROCm 支持。

```python
%%bash
ucx_info -v
```

安装 UCX 后，使用 `ucx_info` 命令检查您的 UCX 库是否是使用 ROCm 支持构建的。在被测Instinct MI300X节点上，`ucx_info -v`显示以下信息：

    库版本：1.20.0
    库路径：/opt/ucx/lib/libucs.so.0
    API 标头版本：1.20.0
    Git 分支“v1.18.1”，修订版 6022e2a
    配置为： --with-rocm=/opt/rocm --enable-mt --prefix=/opt/ucx


<a id="step-33-build-and-install-the-rocm-aware-open-mpi-library"></a>

#### 步骤 3.3：构建并安装 ROCm-Aware Open MPI 库

[Open MPI](https://www.open-mpi.org/) 是一种消息传递接口 (MPI) 实现，用作并行和分布式计算机的通信协议。它具有 ROCm 感知支持，这意味着 MPI 库可以直接从 AMD GPU 设备缓冲区发送和接收数据。 ROCm 支持可通过 UCX 获得，因此在构建 Open MPI 时必须启用 UCX。

**注意**：其他通信传输也可能有效，但 UCX 是 ROCm 设备的 Open MPI 正式支持的唯一传输机制。

```python
%%bash
git clone --recursive https://github.com/open-mpi/ompi.git -b v5.0.x
cd ompi 
./autogen.pl
./configure --prefix=/opt/ompi --with-rocm=/opt/rocm --with-ucx=/opt/ucx
make -j 8
make install 
```

为此 Jupyter 会话设置 Open MPI 和 UCX 环境变量，以便可以发现 `mpirun`、`ompi_info`、`ucx_info` 和共享库。这些更改仅影响当前内核。

```python
import os
os.environ['PATH'] = os.environ['PATH'] + ':/opt/ompi/bin:/opt/ucx/bin'
os.environ['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH'] + ':/opt/ompi/lib:/opt/ucx/lib'
```

通过运行此命令验证 Open MPI ROCm 支持。

```python
%%bash
ompi_info | grep "extensions"
```

安装 Open MPI 后，使用 `ompi_info` 命令验证您的 Open MPI 库是否是使用 ROCm 支持构建的。在用于测试的 Instinct MI300X 设备上运行时，`ompi_info | grep "extensions"` 显示以下信息：

```
MPI extensions: affinity, cuda, ftmpi, rocm
```

<a id="step-34-optional-ssh-passwordless-login-configuration"></a>

#### 步骤3.4：（可选）SSH无密码登录配置

在集群中运行 Open MPI 应用程序时，通常使用 SSH 在远程节点上运行命令来设置分布式 推理。必须在所有远程节点上配置 SSH 无密码登录才能正常工作，该登录允许您无需输入密码或口令即可运行命令。必须在 GPU 集群的每个节点上执行以下步骤。本教程以在两个 GPU 节点上运行命令为例。

1. 在每个节点上使用`ssh-keygen`生成密钥对，由公钥和私钥组成。 `id_rsa` 包含私钥，`id_rsa.pub` 包含公钥。

2. 将本地公钥内容复制到远程节点的`authorized_keys` 文件中。 `authorized_keys` 的文件路径通常为 `~/.ssh/authorized_keys`。 

3. 在每个节点上禁用密码验证。大多数服务器允许用户名/密码身份验证和 SSH 密钥身份验证。要仅允许 SSH 密钥身份验证，请禁用用户名和密码的使用。为此，请取消注释 `/etc/ssh/sshd_config` 中的 `PermitRootLogin prohibit-password` 和 `PubkeyAuthentication yes` 行。 

4. 默认的 SSH 端口为 `22`，该端口可能被集群中的其他 SSH 应用程序占用。为了避免冲突，请更改 Docker 容器内的默认 SSH 端口，该端口仅用于 预填充与解码分离 应用程序。要更改默认端口，请编辑 `/etc/ssh/sshd_config` 并将 `port "self-defined port number"` 添加到文件中，将“自定义端口号”替换为未占用的端口号。

   此外，将相同的自定义端口号添加到 SSH 配置中，以覆盖远程 SSH 连接的默认端口设置。将以下行添加到 `~/.ssh/config` 文件中。

   ```
   Host Remote_node_IP
      Port  "self-defined port number"
   ```

完成上述步骤后，运行以下命令以应用设置。

```python
%%bash 
chmod 600 ~/.ssh/authorized_keys 
chmod 600 ~/.ssh/config 
service ssh restart
```

<a id="step-35-optional-run-the-rccl-test-benchmark-to-test-the-rdma-settings"></a>

#### 步骤 3.5：（可选）运行 RCCL-Test 基准测试来测试 RDMA 设置 

[RCCL-Tests](https://github.com/ROCm/rccl-tests) 是 AMD 提供的开源工具，通过执行 ROCm Collective Communications Library (RCCL) 中的集体操作基准（例如 all_reduce 和 all_gather）来测试 GPU 和节点之间的带宽和 延迟。本教程使用此基准测试工具来验证 RDMA 设备是否已为 LLM 分布式 推理 启用。 

首先，构建 RCCL-Tests 基准。要编译支持 MPI 的 RCCL 测试，请将 `MPI=1` 和 `MPI_HOME` 设置为安装 MPI 的路径。如果 `/opt/rocm` 中未安装 HIP，请在 `HIP_HOME` 中指定位置。同样，如果`/opt/rocm/lib/`中未安装RCCL（`librccl.so`），请将`NCCL_HOME`和`CUSTOM_RCCL_LIB`设置为实际位置。

```python
%%bash
make MPI=1 MPI_HOME=/path/to/mpi HIP_HOME=/path/to/hip NCCL_HOME=/path/to/rccl
```

接下来，在主机文件中指定 GPU 节点。 hostfile 是一个文本文件，其中包含主机（节点）的 IP 地址以及每个节点上可用 GPU 插槽的数量。

第三，运行RCCL基准测试工具。如果节点配置正确，RDMA设备的带宽将远远高于普通以太网设备。

```python
%%bash
TORCH_NCCL_HIGH_PRIORITY=1 RCCL_MSCCL_ENABLE=0 mpirun -np <Total GPU number> --map-by ppr:<GPU number>:node --hostfile <mpi_hosts> --allow-run-as-root --mca pml ucx --mca btl ^openib  -x NCCL_SOCKET_IFNAME=<IP interfaces for communication> -x NCCL_DEBUG=INFO -x NCCL_IB_HCA=<rdma device> -x NCCL_IB_GID_INDEX=3  /home/rccl-tests/build/all_reduce_perf -b 1k -e 2G -f 2 -g 1
```

<a id="run-sglang-pd-disaggregation"></a>

## 运行 SGLang 预填充与解码分离

SGLang 支持 AMD Instinct GPU 上的 预填充-Decode (PD) 分解，它使用 Mooncake 来传输 KV 缓存。从系统架构的角度来看，SGLang 预填充与解码分离 包括 3 个不同的组件：代理服务器、预填充 服务器和解码服务器。当请求到来时，代理服务器根据工作负载平衡方案选择一对 预填充 和解码服务器。所选的 预填充 服务器和解码服务器使用握手配对，分别建立本地发送方和接收方。解码服务器预分配 KV 缓存，然后向 预填充 服务器发出信号以开始 LLM 预填充 推理 并计算 KV 缓存。 预填充 工作完成后，KV 缓存 数据将传输到解码服务器，解码服务器处理迭代令牌生成。

本教程测试 SGLang 预填充与解码分离 的两种配置：节点内 1P1D 和节点间 1P1D。对于节点内情况，您至少需要两个 GPU：一个 GPU 用于运行 预填充 服务器，另一个用于运行解码服务器。对于节点间1P1D，需要两个节点。一个节点将运行 预填充 服务器，另一个节点将运行解码服务器。由于代理服务器不需要大量的GPU资源，因此它运行在预填充节点上。如果您有更大的集群，代理节点可以在独立节点上运行以获得更好的性能。对于演示 SGLang 预填充与解码分离 的以下步骤，示例 预填充 节点的 IP 地址为 `10.21.9.10`，解码节点的地址为 `10.21.9.15`。根据您的集群配置修改相关参数和设置。 

<a id="intra-node-single-node-1p1d"></a>

### 节点内（单节点）1P1D

对于节点内测试，请按照下列步骤操作：

**注意**：从终端运行本节中的所有命令，而不是从Notebook代码单元运行。在 JupyterLab 中，使用 **Launcher → Terminal**（或 **File → New → Terminal**）打开终端。 预填充、解码和代理服务器使用单独的终端。

<a id="run-the-prefill-server"></a>

#### 运行预填充服务器

使用 `sglang.launch_server` 命令启动 预填充 服务器。有关命令选项的更多信息和详细说明，请参阅最新版本的 SGLang 文档或源代码。可以使用 `ibv_devices` 找到 RDMA 设备名称（请参阅前面的部分）。

**注意**：对于多节点配置，请确保 `PATH` 和 `LD_LIBRARY_PATH` 包含 UCX 和 Open MPI（请参阅前面的单元格，或在终端中导出它们）。在运行命令之前替换所有占位符（例如 IP、端口和 RDMA 设备名称）。

```bash
HIP_VISIBLE_DEVICES=0 python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct \
         --disaggregation-mode prefill --port 30000 \
         --disaggregation-ib-device rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
```

<a id="run-the-decode-server"></a>

#### 运行解码服务器

使用 `sglang.launch_server` 命令启动解码服务器。可以使用 `ibv_devices` 找到 RDMA 设备名称。

```bash
HIP_VISIBLE_DEVICES=1 python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct \
        --disaggregation-mode decode --port 30001 \
        --disaggregation-ib-device rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
```

<a id="run-the-proxy-server"></a>

#### 运行代理服务器

此步骤在同一节点上启动代理服务器时配置 预填充 和解码服务器端口。还提供了代理服务器端口，以便测试客户端程序可以连接。

```bash
python -m sglang.srt.disaggregation.mini_lb --prefill http://127.0.0.1:30000 --decode http://127.0.0.1:30001 --host 0.0.0.0 --port 40000 
```

<a id="inter-node-multi-node-1p1d"></a>

### 节点间（多节点）1P1D

对于节点间测试，请按照下列步骤操作：

**注意**：从终端运行本节中的所有命令，而不是从Notebook代码单元运行。在 JupyterLab 中，使用 **Launcher → Terminal**（或 **File → New → Terminal**）打开终端。对 预填充、解码和代理服务器使用单独的终端。

<a id="run-the-etcd-server-on-each-node"></a>

#### 在每个节点上运行etcd服务器

在 预填充 和解码节点的 SGLang ROCm 容器中运行以下命令。这些命令中的etcd服务器端口仅供参考。如果其他进程正在使用它们，请尝试不同的端口。

在预填充节点上，使用以下命令启动etcd服务器。
```bash
etcd --name infra0 --data-dir /var/lib/etcd --initial-advertise-peer-urls http://10.21.9.10:2380 \
  --listen-peer-urls http://10.21.9.10:2380 \
  --listen-client-urls http://10.21.9.10:2379,http://127.0.0.1:2379 \
  --advertise-client-urls http://10.21.9.10:2379 \
  --initial-cluster-token etcd-cluster-1 \
  --initial-cluster infra0=http://10.21.9.10:2380,infra1=http://10.21.9.15:2380 \
  --initial-cluster-state new 
```
在解码节点上，使用此命令运行etcd服务器。
```bash
etcd --name infra1 --data-dir /var/lib/etcd --initial-advertise-peer-urls http://10.21.9.15:2380 \
  --listen-peer-urls http://10.21.9.15:2380 \
  --listen-client-urls http://10.21.9.15:2379,http://127.0.0.1:2379 \
  --advertise-client-urls http://10.21.9.15:2379 \
  --initial-cluster-token etcd-cluster-1 \
  --initial-cluster infra0=http://10.21.9.10:2380,infra1=http://10.21.9.15:2380 \
  --initial-cluster-state new 
```
<a id="run-the-proxy-server"></a>

#### 运行代理服务器

如前所述，在本教程中，该服务器在 预填充 节点上运行。您可以在同一集群中的独立节点上运行它以获得更好的性能。 

本步骤配置预填充和解码节点池的IP地址和端口。还提供代理服务器的IP地址和端口供测试客户端程序连接。 
```bash
python -m sglang.srt.disaggregation.mini_lb --prefill http://10.21.9.10:30000 \
                        --decode http://10.21.9.15:30000 --host 0.0.0.0 --port 40000
```
</details>

<a id="run-the-prefill-server"></a>

#### 运行预填充服务器

使用 `sglang.launch_server` 命令启动 预填充 服务器。有关命令选项的更多信息和详细说明，请参阅最新版本的 SGLang 文档或源代码。可以使用 `ibv_devices` 找到 RDMA 设备名称（请参阅前面的部分）。

```bash
python3 -m sglang.launch_server --model meta-llama/Llama-3.3-70B-Instruct \
                        --disaggregation-mode prefill --disaggregation-ib-device rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7 \
                        --host 10.21.9.10 --port 30000  --trust-remote-code  \
                        --tp 8  --disable-radix-cache --disable-cuda-graph \
                        --max-running-requests 1024 --stream-output \
                        --dist-init-addr 10.21.9.10:5757 --nnodes 1 --node-rank 0 \
                        --mem-fraction-static 0.8
```

<a id="run-the-decode-server"></a>

#### 运行解码服务器

使用 `sglang.launch_server` 命令启动解码服务器。可以使用 `ibv_devices` 找到 RDMA 设备名称。

```bash
python3 -m sglang.launch_server --model meta-llama/Llama-3.3-70B-Instruct \
                        --disaggregation-mode decode --disaggregation-ib-device rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7 \
                        --host 10.21.9.15 --port 30000 --trust-remote-code \
                        --tp 8 --disable-radix-cache --disable-cuda-graph \
                        --max-running-requests 1024 --stream-output \
                        --dist-init-addr 10.21.9.15:5757 --nnodes 1 --node-rank 0 \
                        --mem-fraction-static 0.8
```

<a id="test-pd-disaggregation"></a>

### 测试预填充与解码分离 

在此步骤中，使用 `sglang.bench_serving` 以与普通 SGLang 基准测试相同的方式测试 1P1D 配置。本教程在预填充节点上运行该命令以简化演示。如果要在集群中的另一台机器上运行该命令，请在此命令中设置代理服务器的主机IP地址和端口。其他测试参数可根据需要更改。

```python
%%bash
python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 40000 --dataset-name generated-shared-prefix \
           --gsp-system-prompt-len 0 \
           --gsp-question-len 1024 \
           --gsp-output-len 1024 \
           --gsp-num-groups 1 \
           --gsp-prompts-per-group 16\
           --random-range-ratio 1 \
           --max-concurrency 16 \
           --pd-separated \
           2>&1 | tee test.log
```

<a id="xpyd-setup"></a>

### xPyD 设置

如果您有更大的 GPU 集群来运行 预填充与解码分离，则可以使用 xPyD（多个 预填充 和解码实例）以获得更好的性能。 xPyD 设置与上述步骤相同，并对多节点相关配置进行以下修改：

* 更改代理服务器中的预填充和解码节点配置，例如`--prefill "http://YOUR_FIRST_PREFILL_NODE_IP:30000"`和`--decode "http://YOUR_FIRST_DECODE_NODE_IP:30000"`。
* 启动预填充和解码服务器时，更改多节点分布式服务选项，例如`dist-init-addr`、`nnodes`和`node-rank`。
* 如果需要，更改 SGLang 服务程序的 `tp`、`dp` 和 `ep-size` 选项。

<a id="summary"></a>

## 总结 
在本教程中，您学习了如何在 AMD Instinct MI300X GPU 上设置和运行 SGLang 预填充与解码分离。本教程演示了如何在单个 Instinct MI300X 节点和两个 Instinct MI300X GPU 节点上配置 1P1D，但您可以在自己的 GPU 集群上轻松实现 xPyD。要了解有关 预填充与解码分离 的更多信息，请参阅 [Mooncake](https://kvcache-ai.github.io/Mooncake/)、[LLM-d](https://llm-d.ai/) 和 [vLLM disagg_prefill](https://docs.vllm.ai/en/stable/features/disagg_prefill.html#development) 资源。本教程旨在鼓励您在 AMD GPU 上调整、测试 LLM 分布式 推理 并为其做出贡献，帮助塑造 AI 加速的未来。
