> [Upstream source](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/inference/SGlang_PD_Disagg_On_AMD_GPU.ipynb) · Snapshot: `0071bfd4bd3bd9525405c09a6ae69c983b09312a`

# LLM distributed inference and PD disaggregation on AMD Instinct GPUs

**Author**: Ning Zhang  
**Knowledge level**: Intermediate

With the rapid growth of LLM model sizes, single-node inference optimization starts to show its limitations on LLM serving scaling. Distributed inference on multiple nodes becomes more important for efficient LLM serving. Prefill and Decode (PD) disaggregation is a typical use case for LLM distributed inference on GPU nodes. LLM inference comprises two distinct phases: prefill and decode. The prefill phase is computationally intensive, processing the entire input sequence, while the decode phase is memory-intensive, managing the key-value (KV) cache for token generation. PD disaggregation runs these two phases independently on different GPU nodes, which provides the benefits of efficient GPU resource allocation and independent performance tuning. This tutorial demonstrates how to set up 1P1D distributed inference on either one or two nodes with AMD Instinct™ GPUs. 

## Prerequisites

This tutorial was developed and tested using the following setup.

### Operating system

* **Ubuntu 22.04/24.04**:  Ensure your system is running Ubuntu 22.04 or 24.04.

### Hardware

- **AMD Instinct GPUs**: Tested on MI300X. Works on a single MI300X node or two MI300X nodes (each node has 8 MI300X GPUs).
- **RDMA NIC** (required for single-node and two-node for transferring the KV cache over RDMA):
  - Use an RDMA‑capable NIC (for example, Broadcom Thor2/BCM‑57608 or NVIDIA/Mellanox ConnectX).
  - Install the appropriate vendor driver and RDMA userspace (`rdma-core/libibverbs`). Verify with the following commands:
    - `ibv_devices` and `ibv_devinfo`
    - `ls /dev/infiniband`
  - If running in Docker, ensure `/dev/infiniband` is mapped into the container (see the launch command below).
  - For a two-node configuraton, RDMA must be available on both nodes (RoCEv2 with PFC or InfiniBand) and properly cabled and switch configured.
- **ROCm compatibility**: Use AMD Instinct GPUs with ROCm support and ensure your system meets the [official requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).
### Software

- **ROCm 6.3 or later** (host): Install and verify ROCm by following the [ROCm install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html). After installation, confirm your setup using the `amd-smi` command.

  **Note**: For ROCm 6.4 and earlier, use the `rocm-smi` command instead.
- **Docker** (host): Install the Docker Engine on Linux and verify with the `docker --version` command. You will run this tutorial inside a container. Ensure the host allows you to map GPU and RDMA devices.
  - When launching the container, map the devices and use the recommended flags:
    - Devices: `/dev/kfd`, `/dev/dri`, `/dev/infiniband`, and `/dev/infiniband/rdma_cm`
    - Flags: `--network=host --ipc=host --shm-size 32G --group-add=video`
- **Prebuilt ROCm Docker images** (recommended to reduce the setup effort):
  - [SGLang ROCm image](https://hub.docker.com/r/lmsysorg/sglang/tags)
  - [ROCm Ubuntu 22.04](https://hub.docker.com/r/rocm/dev-ubuntu-22.04)
  - [ROCm Ubuntu 24.04](https://hub.docker.com/r/rocm/dev-ubuntu-24.04)
- **Jupyter (in container)**: Install with `pip install jupyter` to run this notebook.

### Hugging Face API access

* Obtain an API token from [Hugging Face](https://huggingface.co) for downloading models.
* Ensure the Hugging Face API token has the necessary permissions and approvals to access the required checkpoints:
  - For one node, you must have access to the [Meta Llama Llama-3.1-8B-Instruct checkpoints](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).
  - For two nodes, you must have access to the [Meta Llama Llama-3.3-70B-Instruct checkpoints](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct).

## Set up the PD disaggregation environment 

In this tutorial, you will work on the prebuilt ROCm SGLang image, which integrates SGLang with the AMD [ROCm](https://rocm.docs.amd.com/en/latest/index.html) software stack. You can also try other ROCm images as the base image if you want.  

### Step 1: Prepare the tutorial environment

Follow these steps to configure your tutorial environment:

#### Pull the Docker image

Use the `lmsysorg/sglang:v0.4.9-rocm630` Docker image as the base image. This is the latest image that was tested for this tutorial.

**Note**: The SGLang community continues to release additional ROCm SGLang Docker images. You are strongly encouraged to try the latest available image for better performance.

``` bash
docker pull lmsysorg/sglang:v0.4.9-rocm630
```

#### Launch the Docker container

To achieve good network transfer performance, an RDMA NIC is required for the nodes running PD disaggregation. When launching the Docker images, map the RDMA device into the Docker container, as shown in the command below.

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

**Note**: This command mounts the current directory to the `/workspace` directory in the container. Ensure the notebook file is either copied to this directory before running the Docker command or uploaded into the Jupyter Notebook environment after it starts. Save the token or URL provided in the terminal output to access the notebook from your web browser. You can download this notebook from the [AI Developer Hub GitHub repository](https://github.com/ROCm/gpuaidev).

#### Install and launch Jupyter

Inside the Docker container, install Jupyter using the following command:

``` bash
pip install jupyter
```

Start the Jupyter server:

``` bash
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**Note**: Ensure port `8888` is not already in use on your system before running the above command. If it is, you can specify a different port by replacing `--port=8888` with another port number, for example, `--port=8890`.

**Note**: The rest of this notebook is designed to run as a Jupyter notebook. This notebook demonstrates Prefill and Decode (PD) disaggregation on AMD Instinct GPUs. It runs on a single node (Intra‑node 1P1D) by default. The two‑node (Inter‑node 1P1D) steps are optional and clearly marked.

**Run modes**

* Single node (default): The [etcd](https://etcd.io/) key-value store server is not required. Use the "Intra‑node 1P1D" section.
* Two nodes (optional): Requires a setup with the etcd server, RDMA, and SSH. Use the two‑node "Inter‑node 1P1D" section.

#### Provide your Hugging Face token

You'll require a Hugging Face API token to access Llama models with the appropriate permissions as indicated in earlier sections of this notebook. First, install the Hugging Face Hub library.

```python
!pip install --upgrade huggingface_hub
```

Run the following interactive block in your Jupyter notebook to set up the token:

```python
from huggingface_hub import notebook_login, HfApi

# Prompt the user to log in
notebook_login()
```

Verify that your token was accepted correctly:

```python
# Validate the token
try:
    api = HfApi()
    user_info = api.whoami()
    print(f"Token validated successfully! Logged in as: {user_info['name']}")
except Exception as e:
    print(f"Token validation failed. Error: {e}")
```

### Step 2: Install the necessary software components 

For the Intra-node 1P1D mode, you must install the Mooncake transfer engine. The etcd server is not required for this case. For the Inter-node 1P1D mode, you must install both etcd and the Mooncake transfer engine.

#### Step 2.1: (Optional) Install etcd

You can skip this step if you are running the single-node test. etcd is a strongly consistent, distributed key-value store that provides a reliable way to store data that needs to be accessed by a distributed system or cluster of machines. In the SGLang PD disaggregation solution design, the etcd server is required on each GPU node to provide cluster metadata storage. It must be installed to run this tutorial on two nodes.

```python
%%bash
cd /sgl-workspace
apt update && apt install -y wget
wget https://github.com/etcd-io/etcd/releases/download/v3.6.0-rc.5/etcd-v3.6.0-rc.5-linux-amd64.tar.gz -O /tmp/etcd.tar.gz
tar -xvf /tmp/etcd.tar.gz -C /usr/local/bin/ --strip-components=1 && rm /tmp/etcd.tar.gz
```

#### Step 2.2: Install Mooncake

Mooncake features a KV cache-centric disaggregated architecture that separates the prefill and decoding clusters. Its core components, such as the transfer engine, have been integrated into the SGLang PD disaggregation solution to transfer the KV cache between nodes.

```python
%%bash
apt update && apt install -y zip unzip openssh-server
apt -y install gcc make libtool autoconf  librdmacm-dev rdmacm-utils infiniband-diags ibverbs-utils perftest ethtool  libibverbs-dev rdma-core strace
cd /sgl-workspace
pip install mooncake-transfer-engine
```

### Step 3: Install RDMA-related libraries and add the network transfer configuration

For the single-node 1P1D test, you can skip the network configuration in Step 3.4. However, the SGLang/Mooncake transfer engine is still required to enable the RDMA NIC, even for intra-node runs. Verify RDMA availability with `ibv_devices`. If no device is listed, install the appropriate RDMA NIC driver as described in Step 3.1.

#### Step 3.1: Install the NIC RDMA driver 

These instructions were validated on Instinct MI300X systems equipped with Broadcom Thor2/BCM-57608 RDMA NICs (single-node and two-node). If you are using a different RDMA NIC, download and install the appropriate vendor driver according to the documentation. The commands below apply only to the Thor2/BCM-57608 NIC.

⚠️ **Note**: Driver-specific instructions: The commands below are curated for Broadcom Thor2/BCM‑57608 RDMA NICs on Instinct MI300X systems. If your system uses a different RDMA NIC, obtain and install the correct driver and userspace from your NIC vendor following the documentation before proceeding. You can find your NIC vendor by running `lspci | grep -i Eth`.

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

List the RDMA devices.

```python
%%bash
ibv_devices
```

After running the steps above, list the RDMA devices using the `ibv_devices` command and check the RDMA device information using the `ibv_devinfo` command. The output of the `ibv_devices` command should look similar to the following results, confirming that all RDMA devices can be enumerated and are operating successfully.

    device                 node GUID
    ------              ----------------
    rdma0               d604e6fffe0e9fb4
    rdma1               d604e6fffe0e9938
    xeth0               d604e6fffee921d0
    rdma2               d604e6fffe780000
    rdma3               d604e6fffe0e9d34
    rdma4               d604e6fffe780370
    rdma5               d604e6fffe0e8678
    rdma6               d604e6fffe0e8718
    rdma7               d604e6fffe7801f4

#### Step 3.2: Build and install the ROCm-aware UCX library

The [Unified Communication Framework](https://github.com/openucx/ucx) (UCX) is an open-source, cross-platform framework designed to provide a common set of communication interfaces for various network programming models and interfaces. UCX uses ROCm technologies to implement various network operation primitives. It's the standard communication library for InfiniBand and RDMA over Converged Ethernet (RoCE) network interconnection.

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

Set the UCX environment variables for the Jupyter session so that `ucx_info` and the UCX libraries are discoverable. These changes only affect the current kernel.

```python
import os
os.environ['PATH'] = os.environ['PATH'] + ':/opt/ucx/bin'
os.environ['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH'] + ':/opt/ucx/lib'
```

Verify UCX ROCm support by running this command.

```python
%%bash
ucx_info -v
```

After installing UCX, use the `ucx_info` command to check whether your UCX library was built with ROCm support. On the Instinct MI300X node under test, `ucx_info -v` shows the following information:

    Library version: 1.20.0
    Library path: /opt/ucx/lib/libucs.so.0
    API headers version: 1.20.0
    Git branch 'v1.18.1', revision 6022e2a
    Configured with: --with-rocm=/opt/rocm --enable-mt --prefix=/opt/ucx


#### Step 3.3: Build and install the ROCm-Aware Open MPI library

[Open MPI](https://www.open-mpi.org/) is a Message Passing Interface (MPI) implementation used as a communication protocol for parallel and distributed computers. It features ROCm-aware support, which means the MPI library can send and receive data from the AMD GPU device buffers directly. ROCm support is available through UCX, so you must enable UCX when building Open MPI.

**Note**: Other communication transports might also work, but UCX is the only transport mechanism formally supported in Open MPI for ROCm devices.

```python
%%bash
git clone --recursive https://github.com/open-mpi/ompi.git -b v5.0.x
cd ompi 
./autogen.pl
./configure --prefix=/opt/ompi --with-rocm=/opt/rocm --with-ucx=/opt/ucx
make -j 8
make install 
```

Set Open MPI and UCX environment variables for this Jupyter session so `mpirun`, `ompi_info`, `ucx_info`, and the shared libraries are discoverable. These changes affect only the current kernel.

```python
import os
os.environ['PATH'] = os.environ['PATH'] + ':/opt/ompi/bin:/opt/ucx/bin'
os.environ['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH'] + ':/opt/ompi/lib:/opt/ucx/lib'
```

Verify Open MPI ROCm support by running this command.

```python
%%bash
ompi_info | grep "extensions"
```

After installing Open MPI, use the `ompi_info` command to verify whether your Open MPI library was built with ROCm support. When run on the Instinct MI300X device used for the test, `ompi_info | grep "extensions"` displays the following information:

```
MPI extensions: affinity, cuda, ftmpi, rocm
```

#### Step 3.4: (Optional) SSH passwordless login configuration

When running Open MPI applications in a cluster, SSH is typically used to run commands on remote nodes to set up the distributed inference. SSH passwordless login, which lets you run commands without entering a password or passphrase, must be configured on all remote nodes to work properly. The following steps must be performed on each node of the GPU cluster. In this tutorial, the commands are run on two GPU nodes as an example.

1. Use `ssh-keygen` to generate a key pair, consisting of a public key and a private key, on each node. `id_rsa` contains the private key and `id_rsa.pub` contains the public key.

2. Copy the contents of the local public key to the `authorized_keys` file on the remote node. The file path for `authorized_keys` is typically `~/.ssh/authorized_keys`. 

3. Disable password authentication on each node. Most servers allow both username/password authentication and SSH key authentication. To allow only SSH key authentication, disable the use of usernames and passwords. To do so, uncomment the lines `PermitRootLogin prohibit-password` and `PubkeyAuthentication yes` in `/etc/ssh/sshd_config`. 

4. The default SSH Port is `22`, which might be occupied by other SSH applications in the cluster. To avoid conflicts, change the default SSH port inside the Docker container, which is only used for the PD disaggregation application. To change the default port, edit `/etc/ssh/sshd_config` and add `port "self-defined port number"` to the file, replacing "self-defined port number" with an unoccupied port number.

   Additionally, add the same self-defined port number to the SSH configuration to override the default port setting for remote SSH connections. Add the following lines to the  `~/.ssh/config` file.

   ```
   Host Remote_node_IP
      Port  "self-defined port number"
   ```

After completing the steps above, run the commands below to apply the settings.

```python
%%bash 
chmod 600 ~/.ssh/authorized_keys 
chmod 600 ~/.ssh/config 
service ssh restart
```

#### Step 3.5: (Optional) Run the RCCL-Test benchmark to test the RDMA settings 

[RCCL-Tests](https://github.com/ROCm/rccl-tests) is an open-source tool provided by AMD to test the bandwidth and latency between GPUs and nodes by performing the collective operations benchmarks from the ROCm Collective Communications Library (RCCL), such as all_reduce and all_gather. This tutorial uses this benchmarking tool to verify whether the RDMA devices have been enabled for LLM distributed inference. 

First, build the RCCL-Tests benchmark. To compile RCCL tests with MPI support, set `MPI=1` and set `MPI_HOME` to the path where MPI is installed. If HIP is not installed in `/opt/rocm`, specify the location in `HIP_HOME`. Similarly, if RCCL (`librccl.so`) is not installed in `/opt/rocm/lib/`, set `NCCL_HOME` and `CUSTOM_RCCL_LIB` to the actual location.

```python
%%bash
make MPI=1 MPI_HOME=/path/to/mpi HIP_HOME=/path/to/hip NCCL_HOME=/path/to/rccl
```

Next, specify the GPU node in the hostfile. The hostfile is a text file that contains the IP addresses of the hosts (nodes) and the number of available GPU slots on each node.

Third, run the RCCL benchmark tool. If the node configuration is correct, the bandwidth for the RDMA device will be far higher than that of a normal Ethernet device.

```python
%%bash
TORCH_NCCL_HIGH_PRIORITY=1 RCCL_MSCCL_ENABLE=0 mpirun -np <Total GPU number> --map-by ppr:<GPU number>:node --hostfile <mpi_hosts> --allow-run-as-root --mca pml ucx --mca btl ^openib  -x NCCL_SOCKET_IFNAME=<IP interfaces for communication> -x NCCL_DEBUG=INFO -x NCCL_IB_HCA=<rdma device> -x NCCL_IB_GID_INDEX=3  /home/rccl-tests/build/all_reduce_perf -b 1k -e 2G -f 2 -g 1
```

## Run SGLang PD disaggregation

SGLang supports Prefill-Decode (PD) disaggregation on AMD Instinct GPUs, which uses Mooncake to transfer the KV cache. From a system architecture perspective, SGLang PD disaggregation includes 3 distinct components: a proxy server, prefill server, and decode server. When a request comes in, the proxy server selects a pair of prefill and decode servers based on a workload-balancing scheme. The selected prefill server and decode server pair using a handshake, establishing a local sender and receiver, respectively. The decode server preallocates the KV cache, then signals the prefill server to begin LLM prefill inference and compute the KV caches. After the prefill work is done, the KV cache data is transferred to the decode server, which handles iterative token generation.

This tutorial tests SGLang PD Disaggregation for two configurations: Intra-node 1P1D and Inter-node 1P1D. For the Intra-node case, you need at least two GPUs: one GPU to run the prefill server and the other to run the decode server. For inter-node 1P1D, you need two nodes. One node will run the prefill server, and the other node will run the decode server. Because the proxy server doesn't require a large amount of GPU resources, it runs on the prefill node. If you have a larger cluster, the proxy node can run on a standalone node for better performance. For the following steps demonstrating SGLang PD Disaggregation, the example prefill node has the IP address `10.21.9.10` and the decode node has the address `10.21.9.15`. Modify the relevant parameters and settings according to your cluster configuration. 

### Intra-node (single-node) 1P1D

For intra-node testing, follow these steps:

**Note**: Run all commands in this section from a terminal, not from notebook code cells. In JupyterLab, open a terminal using **Launcher → Terminal** (or **File → New → Terminal**). Use separate terminals for the prefill, decode, and proxy servers.

#### Run the prefill server

Use the `sglang.launch_server` command to launch the prefill server. For more information and a detailed description of the command options, see the latest version of the SGLang documentation or source code. The RDMA device names can be found by using `ibv_devices` (see the earlier section).

**Note**: For a multi-node configuration, Ensure `PATH` and `LD_LIBRARY_PATH` include UCX and Open MPI (see the earlier cells, or export them in the terminal). Replace any placeholders (for example, IPs, ports, and RDMA device names) before running the commands.

```bash
HIP_VISIBLE_DEVICES=0 python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct \
         --disaggregation-mode prefill --port 30000 \
         --disaggregation-ib-device rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
```

#### Run the decode server

Use the `sglang.launch_server` command to launch the decode server. The RDMA device names can be found by using `ibv_devices`.

```bash
HIP_VISIBLE_DEVICES=1 python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct \
        --disaggregation-mode decode --port 30001 \
        --disaggregation-ib-device rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
```

#### Run the proxy server

This step configures the prefill and decode server ports when launching the proxy server on the same node. The proxy server port is also provided so the test client program can connect.

```bash
python -m sglang.srt.disaggregation.mini_lb --prefill http://127.0.0.1:30000 --decode http://127.0.0.1:30001 --host 0.0.0.0 --port 40000 
```

### Inter-node (multi-node) 1P1D

For inter-node testing, follow these steps:

**Note**: Run all commands in this section from a terminal, not from notebook code cells. In JupyterLab, open a terminal using **Launcher → Terminal** (or **File → New → Terminal**). Use separate terminals for the prefill, decode, and proxy servers.

#### Run the etcd server on each node

Run the commands below in the SGLang ROCm containers of both the prefill and decode nodes. The etcd server ports in these commands are for reference only. If they are in use by other processes, try different ports.

On the prefill node, start the etcd server using the following command.
```bash
etcd --name infra0 --data-dir /var/lib/etcd --initial-advertise-peer-urls http://10.21.9.10:2380 \
  --listen-peer-urls http://10.21.9.10:2380 \
  --listen-client-urls http://10.21.9.10:2379,http://127.0.0.1:2379 \
  --advertise-client-urls http://10.21.9.10:2379 \
  --initial-cluster-token etcd-cluster-1 \
  --initial-cluster infra0=http://10.21.9.10:2380,infra1=http://10.21.9.15:2380 \
  --initial-cluster-state new 
```
On the decode node, use this command to run the etcd server.
```bash
etcd --name infra1 --data-dir /var/lib/etcd --initial-advertise-peer-urls http://10.21.9.15:2380 \
  --listen-peer-urls http://10.21.9.15:2380 \
  --listen-client-urls http://10.21.9.15:2379,http://127.0.0.1:2379 \
  --advertise-client-urls http://10.21.9.15:2379 \
  --initial-cluster-token etcd-cluster-1 \
  --initial-cluster infra0=http://10.21.9.10:2380,infra1=http://10.21.9.15:2380 \
  --initial-cluster-state new 
```
#### Run the proxy server

As previously mentioned, this server runs on the prefill node in this tutorial. You can run it on a standalone node in the same cluster for better performance. 

In this step, the IP addresses and ports of the prefill and decode node pools are configured. The IP address and port of the proxy server are also provided for the test client program to connect to. 
```bash
python -m sglang.srt.disaggregation.mini_lb --prefill http://10.21.9.10:30000 \
                        --decode http://10.21.9.15:30000 --host 0.0.0.0 --port 40000
```
</details>

#### Run the prefill server

Use the `sglang.launch_server` command to launch the prefill server. For more information and a detailed description of the command options, see the latest version of the SGLang documentation or source code. The RDMA device names can be found by using `ibv_devices` (see the earlier section).

```bash
python3 -m sglang.launch_server --model meta-llama/Llama-3.3-70B-Instruct \
                        --disaggregation-mode prefill --disaggregation-ib-device rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7 \
                        --host 10.21.9.10 --port 30000  --trust-remote-code  \
                        --tp 8  --disable-radix-cache --disable-cuda-graph \
                        --max-running-requests 1024 --stream-output \
                        --dist-init-addr 10.21.9.10:5757 --nnodes 1 --node-rank 0 \
                        --mem-fraction-static 0.8
```

#### Run the decode server

Use the `sglang.launch_server` command to launch the decode server. The RDMA device names can be found by using `ibv_devices`.

```bash
python3 -m sglang.launch_server --model meta-llama/Llama-3.3-70B-Instruct \
                        --disaggregation-mode decode --disaggregation-ib-device rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7 \
                        --host 10.21.9.15 --port 30000 --trust-remote-code \
                        --tp 8 --disable-radix-cache --disable-cuda-graph \
                        --max-running-requests 1024 --stream-output \
                        --dist-init-addr 10.21.9.15:5757 --nnodes 1 --node-rank 0 \
                        --mem-fraction-static 0.8
```

### Test PD disaggregation 

In this step, use `sglang.bench_serving` to test the 1P1D configuration in the same way as a normal SGLang benchmark test. This tutorial runs the command on the prefill node to simplify the demo. To run the command on another machine in the cluster, set the host IP address and port of the proxy server in this command. The other test parameters can be changed as required.

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

### xPyD setup

If you have a larger GPU cluster for running PD disaggregation, you can use xPyD (multiple prefill and decode instances) for better performance. The xPyD setup is the same as the steps described above with the following modifications to the multi-node related configuration:

* Change the prefill and decode node configuration in proxy server, for example, `--prefill "http://YOUR_FIRST_PREFILL_NODE_IP:30000"` and `--decode "http://YOUR_FIRST_DECODE_NODE_IP:30000"`.
* Change the multi-node distributed serving options, such as `dist-init-addr`, `nnodes`, and `node-rank`, when launching the prefill and decode server.
* Change the `tp`, `dp`, and `ep-size` options of the SGLang serving program, if required.

## Summary 
In this tutorial, you learned how to set up and run SGLang PD disaggregation on AMD Instinct MI300X GPUs. The tutorial demonstrated how to configure 1P1D on both a single Instinct MI300X node and two Instinct MI300X GPU nodes, but you can easily implement xPyD on your own GPU cluster. To learn more about PD disaggregation, see the [Mooncake](https://kvcache-ai.github.io/Mooncake/), [LLM-d](https://llm-d.ai/), and [vLLM disagg_prefill](https://docs.vllm.ai/en/stable/features/disagg_prefill.html#development) resources. This tutorial aims to encourage you to tune, test, and contribute to LLM distributed inference on AMD GPUs, helping to shape the future of AI acceleration.
