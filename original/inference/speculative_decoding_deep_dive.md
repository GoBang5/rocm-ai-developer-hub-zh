> [Upstream source](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/inference/speculative_decoding_deep_dive.ipynb) · Snapshot: `0071bfd4bd3bd9525405c09a6ae69c983b09312a`

# Speculative decoding

**Author**: Chang Liu  
**Knowledge level**: Intermediate

This tutorial demonstrates how to achieve an efficiency speedup by enabling speculative decoding in LLM serving. It uses vLLM, one of the most commonly used open-source LLM frameworks, as the serving engine. The tutorial conducts all the benchmark and analysis on AMD Instinct™ MI300X GPUs and the AMD ROCm software stack. For a basic understanding of speculative decoding, including usage guidelines, see the [vLLM Speculative Decoding blog](https://www.amd.com/en/developer/resources/technical-articles/vllm-x-amd-highly-efficient-llm-inference-on-amd-instinct-mi300x-gpus.html). 

vLLM can be up to 2.3 times faster when enabled with speculative decoding. In this tutorial, you'll use Llama-3.1 70B as the base model and Llama-3.1 1B as the draft model, comparing their serving performance with and without speculative decoding enabled. The tutorial provides detailed steps to reproduce the performance gain on AMD MI300X GPUs.


## Prerequisites
This tutorial was developed and tested using the following setup.

### Operating system

* **Ubuntu 22.04**: Ensure your system is running Ubuntu version 22.04.

### Hardware
- **AMD Instinct™ GPU(s)**: This tutorial has been tested on AMD Instinct MI300X GPUs. Ensure you are using an AMD Instinct GPU or compatible hardware with ROCm support and that your system meets the [official requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).


### Software
* **ROCm 6.2+**: Install and verify ROCm by following the [ROCm install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html). After installation, confirm your setup using:

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

* Obtain an API token from [Hugging Face](https://huggingface.co) for downloading models.
* Ensure the Hugging Face API token has the necessary permissions and approval to access the [Meta Llama checkpoints](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).

### Data preparation

This tutorial uses a sample dataset from Hugging Face, which is prepared during the setup steps.


## Preparing the serving environment

### 1. Pull the Docker image

Ensure your system meets the [system requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

Pull the Docker image required for this tutorial:

``` bash
docker pull vllm/vllm-openai-rocm:v0.15.0
```

### 2. Install and launch Jupyter

Inside the Docker container, install Jupyter using the following command:

``` bash
pip install jupyter
```

Start the Jupyter server:

``` bash
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**Note**: Ensure port `8888` is not already in use on your system before running the above command. If it is, you can specify a different port by replacing `--port=8888` with another port number, for example, `--port=8890`.


### 3. Provide your Hugging Face token

You'll require a Hugging Face API token to access Llama-3.1. Generate your token at [Hugging Face Tokens](https://huggingface.co/settings/tokens) and request access for [Llama-3.1 8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct). Tokens typically start with "hf_". 

There are two ways to set up the Hugging Face token: by running a Python script or by running a bash command.

By running a Python script, you can run the following interactive block in your Jupyter notebook to set up the token:

**Note**: Uncheck the "Add token as Git credential" option.

```python
from huggingface_hub import notebook_login, HfApi

# Prompt the user to log in
notebook_login()
```

Verify that your token was accepted correctly. You should see output that begins with "Token validated successfully".

```python
from huggingface_hub import HfApi

try:
    api = HfApi()
    user_info = api.whoami()
    print(f"Token validated successfully! Logged in as: {user_info['name']}")
except Exception as e:
    print(f"Token validation failed. Error: {e}")
```

## Downloading the model weights (Optional)

You can download the model weights in `\models` in advance using the commands below. This tutorial uses the AMD-optimized Llama-3.1 70B as the base model and Llama-3.1 1B as the speculative draft model.

**Note**: This step is optional. If you choose to skip it, the models used in the LLM serving will be automatically downloaded when the serving process starts, based on your specified commands.

```python
# Download Llama-3.1 70B Instruct FP8 KV model weights in local
!hf download \
  amd/Llama-3.1-70B-Instruct-FP8-KV \
  --local-dir /models/amd--Llama-3.1-70B-Instruct-FP8-KV

# Download Llama-3.1 1B Instruct FP8 KV model weights in local
!hf download \
  amd/Llama-3.2-1B-Instruct-FP8-KV \
  --local-dir /models/amd--Llama-3.2-1B-Instruct-FP8-KV
```

## Starting the serving

You can use the online serving mode of vLLM to compare the performance difference between disabling and enabling speculative decoding.

Use the command below to start the server without speculative decoding enabled. Modify the port if port 8001 is in use:

```python
%%bash
export MODEL_PATH=/path/to/model/weights/cache/directory/
export WORK_PATH=/path/to/workspace/
docker run -d --rm \
  --network=host \
  --ipc=host \
  --privileged \
  --cap-add=CAP_SYS_ADMIN \
  --device=/dev/kfd \
  --device=/dev/dri \
  --device=/dev/mem \
  --group-add render \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --name vllm_spec_dec \
  -v $MODEL_PATH:/models \
  -v $WORK_PATH:/work \
  vllm/vllm-openai-rocm:v0.15.0 \
  bash -c "PYTORCH_TUNABLEOP_ENABLED=0 PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS=100 PYTORCH_TUNABLEOP_MAX_WARMUP_DURATION_MS=10 PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE=1024 PYTORCH_TUNABLEOP_FILENAME=afo_tune_device_%d_full.csv HIP_FORCE_DEV_KERNARG=1 VLLM_USE_ROCM_CUSTOM_PAGED_ATTN=1 VLLM_INSTALL_PUNICA_KERNELS=1 TOKENIZERS_PARALLELISM=false RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1 NCCL_MIN_NCHANNELS=112 VLLM_USE_TRITON_FLASH_ATTN=0 VLLM_FP8_PADDING=1 VLLM_FP8_ACT_PADDING=1 VLLM_FP8_WEIGHT_PADDING=1 VLLM_FP8_REDUCE_CONV=1 \
  vllm serve /models/amd--Llama-3.1-70B-Instruct-FP8-KV \
    --swap-space 16 \
    --disable-log-requests \
    --tensor-parallel-size 1 \
    --distributed-executor-backend mp \
    --dtype float16 \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --enable-chunked-prefill=False \
    --max-num-seqs 300 \
    --port 8001"
```

**Note**: This command mounts `$MODEL_PATH` to the `/models` directory in the container and `$WORK_PATH` to the `/work` directory in the container. Ensure the notebook file is copied to your `$WORK_PATH` directory before running the Docker command or uploaded into the Jupyter Notebook environment after it starts. Save the token or URL provided in the terminal output to access the notebook from your web browser. You can download this notebook from the [AI Developer Hub GitHub repository](https://github.com/ROCm/gpuaidev).

The script needs to provide enough time for the `vllm serve` command to launch the LLM service. The bigger LLMs require more loading time, so you might need to adjust the sleep time for your environment. Another method is to start the two vLLM server containers before running the subsequent steps of this Jupyter notebook, then use a `curl` test to ensure the `vllm serve` command is finished.

```python
!sleep 360
```

Now test whether the two `vllm serve` operations for the containers are ready. If not, you should add more time by using the `sleep` command in bash until you receive the correct response from the `curl` test.

```python
!curl http://localhost:8001/v1/models
```

## Setting the client

After starting the server, start a client to send out requests. For the client, use the SGLang Docker image `lmsysorg/sglang:v0.5.8-rocm700-mi30x`. Launch another terminal in your environment, initialize the client Docker container, and benchmark the performance using the command below:

```python
%%bash
docker run -it --rm \
  --ipc=host \
  --network=host \
  --privileged \
  --cap-add=CAP_SYS_ADMIN \
  --device=/dev/kfd \
  --device=/dev/dri \
  --device=/dev/mem \
  --group-add render \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size=192g \
  --name spec_dec_client \
  -v $MODEL_PATH:/models \
  -v $WORK_PATH:/work \
  lmsysorg/sglang:v0.5.8-rocm700-mi30x \
  bash -c "python3 -m sglang.bench_serving \
    --backend vllm \
    --dataset-name random \
    --num-prompt 500 \
    --request-rate 1.0 \
    --random-input 8192 \
    --port 8001 \
    --random-output 256 > /work/vllm_base_log"
```

**Note**: Like you did for the command for starting the server, modify `$MODEL_PATH` and `$WORK_PATH` to correspond to your actual model directory and workspace.

After the client command runs, it generates a log file named `vllm_base_log`.

## Enabling speculative decoding in serving

Before running the cell below, stop the previous cell that was running the server without speculative decoding enabled.
Use the command below to start the server with speculative decoding enabled. Modify the port if port 8001 is in use:

```python
docker run -d --rm \
  --network=host \
  --ipc=host \
  --privileged \
  --cap-add=CAP_SYS_ADMIN \
  --device=/dev/kfd \
  --device=/dev/dri \
  --device=/dev/mem \
  --group-add render \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --name vllm_spec_dec \
  -v $MODEL_PATH:/models \
  -v $WORK_PATH:/work \
  vllm/vllm-openai-rocm:v0.15.0 \
  bash -c "PYTORCH_TUNABLEOP_ENABLED=0 \
  PYTORCH_TUNABLEOP_TUNING=0 \
  PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS=100 \
  PYTORCH_TUNABLEOP_MAX_WARMUP_DURATION_MS=10 \
  PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE=1024 \
  PYTORCH_TUNABLEOP_FILENAME=afo_tune_device_%d_full.csv \
  HIP_FORCE_DEV_KERNARG=1 \
  VLLM_USE_ROCM_CUSTOM_PAGED_ATTN=1 \
  VLLM_INSTALL_PUNICA_KERNELS=1 \
  TOKENIZERS_PARALLELISM=false \
  RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1 \
  NCCL_MIN_NCHANNELS=112 \
  VLLM_USE_TRITON_FLASH_ATTN=0 \
  VLLM_FP8_PADDING=1 \
  VLLM_FP8_ACT_PADDING=1 \
  VLLM_FP8_WEIGHT_PADDING=1 \
  VLLM_FP8_REDUCE_CONV=1 \
  vllm serve /models/amd--Llama-3.1-70B-Instruct-FP8-KV \
    --swap-space 16 \
    --disable-log-requests \
    --tensor-parallel-size 1 \
    --distributed-executor-backend mp \
    --dtype float16 \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --enable-chunked-prefill=False \
    --max-num-seqs 300 \
    --port 8001 \
    --speculative-model /models/amd--Llama-3.2-1B-Instruct-FP8-KV \
    --num_speculative_tokens 5 \
    --speculative-model-quantization fp8"
```

The script needs to provide enough time for the `vllm serve` command to launch the LLM service. The bigger LLMs require more loading time, so you might need to adjust the sleep time for your environment. Another method is to start the two vLLM server containers before running the subsequent steps of this Jupyter notebook, then use a `curl` test to ensure the `vllm serve` command is finished.

```python
!sleep 360
```

Now test whether the two `vllm serve` operations for the containers are ready. If not, you should add more time by using the `sleep` command in bash until you receive the correct response from the `curl` test.

```python
!curl http://localhost:8001/v1/models
```

## Sending requests

After starting the server, use the same client Docker container `spec_dec_client` to send out requests. Please ensure the previous container has stopped before starting. For the client container, use the SGLang Docker image `lmsysorg/sglang:v0.5.8-rocm700-mi30x`.

In the terminal, use the command below to send requests:

```python
%%bash
docker run -d --rm \
  --ipc=host \
  --network=host \
  --privileged \
  --cap-add=CAP_SYS_ADMIN \
  --device=/dev/kfd \
  --device=/dev/dri \
  --device=/dev/mem \
  --group-add render \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size=192g \
  --name spec_dec_client \
  -v $MODEL_PATH:/models \
  -v $WORK_PATH:/work \
  lmsysorg/sglang:v0.5.8-rocm700-mi30x \
  bash -c "python3 -m sglang.bench_serving \
    --backend vllm \
    --dataset-name random \
    --num-prompt 500 \
    --request-rate 1.0 \
    --random-input 8192 \
    --port 8001 \
    --random-output 256 > /work/vllm_spec_dec_log"
```

To explore speculative decoding more thoroughly, you can modify the value of each parameter (for example, `request-rate`) in the above command to observe the performance changes. For more information, see the study in the [Speculative Decoding - Deep Dive](https://rocm.blogs.amd.com/software-tools-optimization/speculative-decoding---deep-dive/README.html) blog.
