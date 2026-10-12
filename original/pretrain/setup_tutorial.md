> [Upstream source](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/pretrain/setup_tutorial.ipynb) · Snapshot: `0071bfd4bd3bd9525405c09a6ae69c983b09312a`

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

### 2. Pull the Docker image

Run the following command in your terminal to pull the prebuilt Docker image. The Docker image contains all the necessary dependencies, including PyTorch, PyTorch Lightning, the ROCm libraries, and Megatron-LM utilities.

``` bash
docker pull rocm/megatron-lm:24.12-dev
```

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

### 5. Clone the Megatron-LM repository
Run the following commands inside the Docker container to clone the Megatron-LM repository and navigate to the validated commit:

```python
# Clone the Megatron-LM repository and navigate to the validated commit
!git clone https://github.com/ROCm/Megatron-LM && cd Megatron-LM && git checkout bb93ccbfeae6363c67b361a97a27c74ab86e7e92
```

## Preparing the training dataset

While the next tutorial uses "mock data" to simplify the tutorial, this section covers an example showing how to preprocess your data for training if you don’t have preprocessed data. This section uses the [BookCorpus dataset](https://huggingface.co/datasets/bookcorpus/bookcorpus), which is a collection of books that has been used for training language models. It contains diverse and continuous text passages, making it suitable for pretraining tasks.

In this tutorial, you will:

* Download and inspect the BookCorpus dataset.
* Convert it to JSONL format.
* Download the necessary tokenizer files (`vocab.json` and `merges.txt`).
* Preprocess the data for training a large language model with Megatron-LM.

### 1. Download and inspect the BookCorpus dataset

Use the Hugging Face datasets library to download the BookCorpus dataset. This step ensures you have access to the raw data needed for preprocessing.

```python
from datasets import load_dataset

# Load BookCorpus dataset
dataset = load_dataset("bookcorpus/bookcorpus", trust_remote_code=True, split="train")

# Inspect the dataset
print("Dataset Structure:", dataset)
print("Sample Data:", dataset[0])  # Access the first record
```

### 2. Convert to the JSONL format

Megatron-LM's preprocessing script requires that the input be in JSONL format, where each line represents a document as a JSON object. This step converts the dataset into the required format.

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

Before moving to the next step, ensure that the dataset is correctly converted to JSONL format. Inspect the first few lines to confirm the structure.

```python
# Inspect the first few lines of the JSONL file
with open(output_file, "r") as f:
    for i in range(5):  # Print the first 5 lines
        print(json.loads(f.readline()))
```

### 3. Download the tokenizer files

Decide which tokenizer you are going to use for your training. As an example, you might pick the GPT-2 Byte Pair Encoding (BPE) tokenizer. The preprocessing tool provided by Megatron requires some additional files for each tokenizer. In the case of BPE, download the following two files that define the tokenizer rules:

* `vocab.json`: Maps the tokens to unique IDs.
* `merges.txt`: Specifies how subword units are combined into tokens.

Download these files to tokenize the dataset correctly.

```python
# Download vocab.json and merges.txt for GPT-2 tokenizer
!wget https://huggingface.co/gpt2/resolve/main/vocab.json -O vocab.json
!wget https://huggingface.co/gpt2/resolve/main/merges.txt -O merges.txt
```

### 4. Preprocess the data

This step tokenizes the dataset and converts it into binary and index files suitable for training a large language model using Megatron-LM. Use the Megatron-LM preprocessing script with the converted JSONL dataset and tokenizer files.

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

**Note:** You might need to modify these parameters based on the tokenizer you pick and your dataset. 

Now check the output files generated during preprocessing to ensure everything is processed correctly and ready for training.

```python
# List output files
!ls output/
```

### 5. Configure the network interfaces (only applicable for multi-node training)

You must set the `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` variables. While this is easily accomplished using the `export` command, this notebook can automatically set the network interface variables.

First, install the `iproute2` package by running the following command:

```python
!apt install -y iproute2
```

Then run the following commands to automatically detect the active network interfaces and set the environment variables based on the first available interface:

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

After running the commands, verify the active network interface using the following command:

```python
!ip a
```

Ensure the detected interface matches your system's active network interface. If necessary, modify the script above or manually set the `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` variables. You can manually set these variables using the directives below:

``` bash
export NCCL_SOCKET_IFNAME=<network_interface>
export GLOO_SOCKET_IFNAME=<network_interface>
```

**Important**: For multi-node training, ensure the following requirements are met:

* Step 5 must be executed on **all nodes** participating in the training.  
* The environment variables `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` must be set to the **same value** across all nodes to ensure consistent network communication.

## Training configuration

Before launching a training task, it's crucial to configure the training environment properly. This section covers the essential configurations for running pretraining, including:

* The training mode (single or multi-node training)
* Dataset options
* Tokenizer selection

### Training modes

The two training modes are single-node training and multi-node training.

#### Single-node training

In single-node training, all computations occur on a single server or machine with multiple GPUs. This is simpler to set up and sufficient for smaller-scale experiments.

To launch training on a single node, use the following command:

```python
!TEE_OUTPUT=1 MBS=2 BS=64 TP=8 TE_FP8=0 SEQ_LENGTH=4096 bash examples/llama/train_llama2.sh
```

The variables in the previous command are:

* `TEE_OUTPUT=1`: Enables output streaming to all nodes.
* `MBS=2`: Sets the micro-batch size.
* `BS=64`: Sets the global batch size.
* `TP=8`: Configures tensor parallelism with 8-way parallelism.
* `TE_FP8=0`: Disables FP8 optimizations (set this to `1` to enable).
* `SEQ_LENGTH=4096`: Specifies the maximum sequence length for training.

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

```python
!TEE_OUTPUT=1 MBS=2 BS=64 TP=8 TE_FP8=0 SEQ_LENGTH=4096 bash examples/llama/train_llama2.sh
```

**Important:** For multi-node training, ensure you run all the previous steps on all the nodes and modify the `NODE_RANK` on each node as described above.

**Tip**: Test multi-node communication with a mock training task before launching full-scale training. This helps debug any node communication or dataset path issues.

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

### Tokenizer selection

Tokenization is the process of converting raw text into tokens that the model can process. Different Llama models require specific tokenizers:

#### For Llama-2 models:

Use the `Llama2Tokenizer`.

#### For Llama-3 and Llama-3.1 models:

Use the `HuggingFaceTokenizer`. Set the Hugging Face model link in the `TOKENIZER_MODEL` variable. For example:

``` bash
TOKENIZER_MODEL=meta-llama/Llama-3.1-8B
```

## Next steps

Proceed to the [Training Llama-3.1 8B with Megatron-LM](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/train_llama_mock_data.html) guide, where you will:

* Use the environment and configurations set up in this tutorial.
* Run practical pretraining examples using mock data.
