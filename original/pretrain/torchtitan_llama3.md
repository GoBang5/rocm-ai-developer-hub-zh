> [Upstream source](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/pretrain/torchtitan_llama3.ipynb) · Snapshot: `0071bfd4bd3bd9525405c09a6ae69c983b09312a`

# Pretrain Llama-3.1 8B with torchtitan

**Authors**: Xinyu Kang and Liz Li  
**Knowledge level**: Intermediate

This tutorial demonstrates how to pretrain the Llama-3.1 8B large language model (LLM) on AMD ROCm GPUs by leveraging torchtitan. Torchtitan is a proof of concept for large-scale LLM training using native PyTorch. The repository showcases PyTorch's latest distributed training features in a clean, minimal codebase. Torchtitan is complementary to, and not a replacement for, other large-scale LLM training codebases such as Megatron, MegaBlocks, LLM Foundry, and DeepSpeed. For more information, see the [official torchtitan GitHub page](https://github.com/pytorch/torchtitan).

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

Ensure your system meets the [System Requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

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
!pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm6.3/ --force-reinstall
!pip3 install ipywidgets
!git clone https://github.com/pytorch/torchtitan
!cd torchtitan && pip install -r requirements.txt && pip install .
# This note book is verified under torch==2.7.0.dev20250302+rocm6.3,  torchvision==0.22.0.dev20250302+rocm6.3, torchtitan==0.0.2
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
torchtitan              0.0.2

```

**⚠️ Important: ensure the correct kernel is selected**

If the verification process fails, ensure the correct Jupyter kernel is selected for your notebook.
To change the kernel, follow these steps:

1. Go to the **Kernel** menu.
2. Select **Change Kernel**.
3. Select `Python 3 (ipykernel)` from the list.

**Important**: Failure to select the correct kernel can lead to unexpected issues when running the notebook.

### 5. Provide your Hugging Face token

You'll require a Hugging Face API token to access Llama-3.1. Generate your token at [Hugging Face Tokens](https://huggingface.co/settings/tokens) and request access for [Llama-3.1 8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B). Tokens typically start with "hf_". 

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

## Pretrain the model

This section covers the process of setting up and running pretraining for the Llama-3.1 model using torchtitan. The following steps describe how to set up GPUs, import the required libraries, configure the model and training parameters, and run the pretraining process.

**⚠️ Important: ensure the correct kernel is selected**

Ensure the correct Jupyter kernel is selected for your notebook. To change the kernel, follow these steps:

1. Go to the **Kernel** menu.
2. Select **Change Kernel**.
3. Select `Python 3 (ipykernel)` from the list.

**Important**: Failure to select the correct kernel can lead to unexpected issues when running the notebook.

### Set and verify the GPU availability

Begin by specifying the GPUs that are available for pretraining. Verify that they are properly detected by PyTorch.

```python
import os
import torch
gpus = [0, 1, 2, 3, 4, 5, 6, 7] # Specify the 8 GPUs to be used for training for MI300x
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ','.join(map(str, gpus)))
# Ensure PyTorch detects the GPUs correctly
print(f"PyTorch detected number of available devices: {torch.cuda.device_count()}") 
```

### Download the Llama tokenizer

 To start training Llama-3.1 models, you must download a tokenizer model. Run the following command to download the Llama-3.1 tokenizer to your local machine.

```python
# Get your HF token from https://huggingface.co/settings/tokens

# Llama-3.1 tokenizer model
!cd torchtitan && python scripts/download_tokenizer.py --repo_id meta-llama/Meta-Llama-3.1-8B --tokenizer_path "original"
```

### Prepare the dataset

 By default, torchtitan Llama-3.1 8B uses the [c4 dataset](https://huggingface.co/datasets/allenai/c4) to conduct pretraining. In this notebook, you can change this to the smaller c4_test dataset for test purposes.

```python
!sed -i 's/dataset = "c4"/dataset = "c4_test"/' ./torchtitan/torchtitan/models/llama/train_configs/llama3_8b.toml
```

### Run the pretrain recipe

You can pretrain Llama-3.1 8B with LoRA on eight GPUs using the following command:

```python
!cd ./torchtitan/ && CONFIG_FILE="./torchtitan/models/llama/train_configs/llama3_8b.toml" ./run_train.sh
```

For multinode distributed training, torchtitan provides a Slurm recipe. You can use the `multinode_trainer.slurm` file to submit your `sbatch` job. For more information, see the official [guide](https://github.com/pytorch/torchtitan?tab=readme-ov-file#multi-node-training) from torchtitan.

### Monitoring GPU memory

To monitor GPU memory during training, run the following command in a terminal.

**Note**: For ROCm 6.4 and earlier, use the `rocm-smi` command instead.

```python
!amd-smi
```

This command displays memory usage and other GPU metrics to ensure your hardware resources are being optimally used.
