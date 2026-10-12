> [Upstream source](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/fine_tune/torchtune_llama3.ipynb) · Snapshot: `0071bfd4bd3bd9525405c09a6ae69c983b09312a`

# Fine-tune Llama-3.1 8B with torchtune

**Author**: Bill He  
**Knowledge level**: Intermediate

This tutorial demonstrates how to fine-tune the Llama-3.1 8B large language model (LLM) on AMD ROCm GPUs by leveraging torchtune. Torchtune is an easy-to-use PyTorch library for authoring, post-training, and experimenting with LLMs. It features:

- Hackable training recipes for SFT, knowledge distillation, RL and RLHF, and quantization-aware training.
- Simple PyTorch implementations of popular LLMs like Llama, Gemma, Mistral, Phi, Qwen, and more.
- OOTB best-in-class memory efficiency, performance improvements, and scaling, utilizing the latest PyTorch APIs.
- YAML configs to easily configure training, evaluation, quantization, or inference recipes.

For more information, see the [official torchtune GitHub page](https://github.com/pytorch/torchtune).

## Prerequisites

This tutorial was developed and tested using the following setup. 

### Operating system

* **Ubuntu 22.04**: Ensure your system is running Ubuntu version 22.04.

### Hardware

* **AMD Instinct™ GPUs**: This tutorial was tested on an AMD Instinct MI300X GPU. Ensure you are using an AMD Instinct GPU or compatible hardware with ROCm support and that your system meets the [official requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

### Software

* **ROCm 6.3**: Install and verify ROCm by following the [ROCm install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html). After installation, confirm your setup using:

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

* This tutorial uses a sample dataset from Hugging Face, which is prepared during the setup steps.

## Prepare the training environment

### 1. Pull the Docker image

Ensure your system meets the [system requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

Pull the Docker image required for this tutorial:

``` bash
docker pull rocm/pytorch-training:latest
```

### 2. Launch the Docker container

Launch the Docker container and map the necessary directories. Replace `/path/to/notebooks` with the full path to the directory on your host machine where these notebooks are stored.

``` bash
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
  rocm/pytorch-training:latest
```

**Note**: This command mounts the current directory to the `/workspace` directory in the container. Ensure the notebook file is either copied to this directory before running the Docker command or uploaded into the Jupyter Notebook environment after it starts. Save the token or URL provided in the terminal output to access the notebook from your web browser. You can download this notebook from the [AI Developer Hub GitHub repository](https://github.com/ROCm/gpuaidev).

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

### 4. Install the required libraries

Install the libraries required for this tutorial. Run the following commands inside the Jupyter notebook running within the Docker container:

```python
# Install PyTorch, torchvision, torchao nightlies
!pip install --pre --upgrade torch torchvision torchao --index-url https://download.pytorch.org/whl/nightly/rocm6.3/
!pip install --pre --upgrade torchtune --extra-index-url https://download.pytorch.org/whl/nightly/rocm6.3/ 
# This note book is verified under torch==2.7.0.dev20250302+rocm6.3, torchao==0.10.0.dev20250303+rocm6.3,  torchvision==0.22.0.dev20250302+rocm6.3, torchtune==0.6.0.dev20250302+rocm6.3
```

Verify the installation:

```python
# Verify the installation and version of the required libraries
!pip list | grep torch
```

Here is the expected output:

```
pytorch-triton-rocm     3.2.0+git4b3bb1f8
torch                   2.7.0.dev20250302+rocm6.3
torchdata               0.11.0
torchtune               0.6.0.dev20250302+rocm6.3
```

### 5. Verify torchtune for ROCm 6.3

To confirm that the package is installed correctly, run the following command:

```python
!tune --help
```

You should see the following output:

```
usage: tune [-h] {download,ls,cp,run,validate,cat} ...

Welcome to the torchtune CLI!

options:
  -h, --help            show this help message and exit

subcommands:
  {download,ls,cp,run,validate,cat}
    download            Download a model from the Hugging Face Hub or Kaggle
                        Model Hub.
    ls                  List all built-in recipes and configs
    cp                  Copy a built-in recipe or config to a local path.
    run                 Run a recipe. For distributed recipes, this supports
                        all torchrun arguments.
    validate            Validate a config and ensure that it is well-formed.
    cat                 Pretty print a config, making it easy to know which
                        parameters you can override with `tune run`.
```

**⚠️ Important: ensure the correct kernel is selected**

If the verification process fails, ensure the correct Jupyter kernel is selected for your notebook.
To change the kernel, follow these steps:

1. Go to the **Kernel** menu.
2. Select **Change Kernel**.
3. Select `Python 3 (ipykernel)` from the list.

**Important**: Failure to select the correct kernel can lead to unexpected issues when running the notebook.

### 6. Provide your Hugging Face token

You'll require a Hugging Face API token to access Llama-3.1. Generate your token at [Hugging Face Tokens](https://huggingface.co/settings/tokens) and request access for [Llama-3.1 8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct). Tokens typically start with "hf_". 

Run the following interactive block in your Jupyter notebook to set up the token:

**Note**: Uncheck the "Add token as Git credential" option.

```python
from huggingface_hub import notebook_login, HfApi

# Prompt the user to log in
notebook_login()
```

Verify that your token was accepted correctly:

```python
from huggingface_hub import HfApi

try:
    api = HfApi()
    user_info = api.whoami()
    print(f"Token validated successfully! Logged in as: {user_info['name']}")
except Exception as e:
    print(f"Token validation failed. Error: {e}")
```

## Fine-tuning the model

This section covers the process of setting up and running fine-tuning for the Llama-3.1 model using torchtune. The following steps describe how to set up GPUs, import the required libraries, configure the model and training parameters, and run the fine-tuning process.

**⚠️ Important: ensure the correct kernel is selected**

Ensure the correct Jupyter kernel is selected for your notebook. To change the kernel, follow these steps:

1. Go to the **Kernel** menu.
2. Select **Change Kernel**.
3. Select `Python 3 (ipykernel)` from the list.

**Important**: Failure to select the correct kernel can lead to unexpected issues when running the notebook.

### Set and verify the GPU availability

Begin by specifying the GPUs available for fine-tuning. Verify that they are properly detected by PyTorch.

```python
import os
import torch
gpus= [0, 1] # Rank 0 is for MI300x single device finetune, and Rank 0/1 for full 
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ','.join(map(str, gpus)))
# Ensure PyTorch detects the GPUs correctly
print(f"PyTorch detected number of available devices: {torch.cuda.device_count()}") 
```

### Download the Llama model

Run the following command to download the weights to your local machine. This also downloads the tokenizer model and a responsible use guide.

To download Llama-3.1, use the following command:

```python
!tune download meta-llama/Meta-Llama-3.1-8B-Instruct \
    --output-dir /tmp/Meta-Llama-3.1-8B-Instruct \
    --ignore-patterns "original/consolidated.00.pth"
```

### Run the fine-tuning recipes

Use these fine-tuning recipes for single GPU or distributed training.

#### Single GPU training

By default, the [alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset is used for fine-tuning, You can fine-tune Llama-3.1 8B with LoRA on a single GPU using the following command:

```python
!tune run lora_finetune_single_device --config llama3_1/8B_lora_single_device
```

#### Distributed training
For distributed training, the tune CLI integrates with torchrun.

**Note**: you must have more than one GPU available to run the distributed training example. To run a full fine-tune of Llama-3.1 8B on two GPUs, use this command:

```python
!tune run --nproc_per_node 2 full_finetune_distributed --config llama3_1/8B_full
```

### Customize your recipes for Llama

There are two ways to modify configurations.

#### Configuration overrides

You can directly overwrite configuration fields from the command line. For example, you can set the batch size to `16` and disable activation checkpoints using the same command:

```python
!tune run lora_finetune_single_device \
    --config llama3_1/8B_lora_single_device \
    batch_size=16 \
    enable_activation_checkpointing=False 
```

**Note**: If you encounter out-of-memory (OOM) errors, reduce the `batch_size` or enable gradient checkpointing. Use `amd-smi` to monitor VRAM usage during fine-tuning. For ROCm 6.4 and earlier, use the `rocm-smi` command for this purpose.

#### Update a local copy

You can also copy the configuration to your local directory and modify the contents directly:

```python
!tune cp llama3_1/8B_lora_single_device ./my_custom_config_llama3_1_8B_lora_single_device.yaml
# Copied to ./my_custom_config_llama3_1_8B_lora_single_device.yaml
```

Then you can run your custom recipe by applying the `tune run` command to your local files:

```python
!tune run lora_finetune_single_device --config ./my_custom_config_llama3_1_8B_lora_single_device.yaml
```

Run the `tune --help` command to see all possible CLI commands and options. For more information on using and updating configurations, see the official torchtune [deep-dive](https://meta-pytorch.org/torchtune/main/deep_dives/configs.html).

### Custom datasets

`torchtune` supports fine-tuning on a variety of different datasets, including instruct-style, chat-style, preference datasets, and more. To learn more about how to apply these components to fine-tune on your own custom dataset, see the provided links along with the torchtune API [docs](https://meta-pytorch.org/torchtune/main/api_ref_datasets.html).

### Monitoring GPU memory

To monitor GPU memory during training, run the following command in a terminal.

**Note**: For ROCm 6.4 and earlier, use the `rocm-smi` command instead.

```python
!amd-smi
```

This command displays memory usage and other GPU metrics to ensure your hardware resources are being optimally used.
