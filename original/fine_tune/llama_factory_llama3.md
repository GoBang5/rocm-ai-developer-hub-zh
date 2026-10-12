> [Upstream source](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/fine_tune/llama_factory_llama3.ipynb) · Snapshot: `0071bfd4bd3bd9525405c09a6ae69c983b09312a`

# Fine-tune Llama-3-8B with Llama-Factory

**Author**: Alex He  
**Knowledge level**: Intermediate

[![Open in AMD Developer Cloud](../assets/jupyter_notebook_button_fnl_small.png)](https://notebooks.amd.com/tutorials/llama_factory_llama3.ipynb)

This tutorial demonstrates how to fine-tune the Llama-3 8B large language model (LLM) on AMD ROCm GPUs by leveraging Llama-Factory. Efficient fine-tuning is vital for adapting large language models (LLMs) to downstream tasks. However, non-trivial efforts are required to implement these methods on different models.

[Llama-Factory](https://github.com/hiyouga/LLaMA-Factory) is a unified framework that integrates a suite of cutting-edge, efficient training methods.

It provides a solution for flexibly customizing the fine-tuning of over 100 LLMs without the need for coding through the built-in web UI LLAMABOARD.

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

This tutorial uses a sample dataset from Hugging Face, which is prepared during the setup steps.

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

### 4. Install the Required Libraries and Frameworks

Run the cell below to install the libraries and frameworks required for this tutorial.

* **DeepSpeed** – Optimizes large-model training by reducing memory usage and enabling efficient distributed training.
* **LLaMA-Factory** – An open-source framework that simplifies the training and fine-tuning of Large Language Models (LLMs).

```python
!pip3 install -U deepspeed==0.16.5
# Install Llama-Factory from source, pinned to commit a0d44c65 (0.9.4.dev0):
# the first commit with examples/deepspeed/, and still requires-python >=3.9
# so it installs on this container's Python 3.10.
!rm -rf LLaMA-Factory
!mkdir LLaMA-Factory && cd LLaMA-Factory && git init -q && git remote add origin https://github.com/hiyouga/LLaMA-Factory.git && git fetch -q --depth 1 origin a0d44c650a9a38c1ec6c8fc48a7daeacca9900a9 && git checkout -q FETCH_HEAD
!cd LLaMA-Factory && pip install -e ".[torch,metrics]"
# This notebook is verified under llamafactory==0.9.3.dev0 deepspeed==0.16.5
```

Verify the installation:

```python
# Verify the installation and version of the required libraries
!pip list | grep llamafactory
```

Here is the expected output:

```
llamafactory            0.9.4.dev0 
```

### 5. Verify Llama-Factory 

To confirm the package is installed correctly, run the following command:

```python
!llamafactory-cli help
```

You should see the following output:

```
----------------------------------------------------------------------
| Usage:                                                             |
|   llamafactory-cli api -h: launch an OpenAI-style API server       |
|   llamafactory-cli chat -h: launch a chat interface in CLI         |
|   llamafactory-cli eval -h: evaluate models                        |
|   llamafactory-cli export -h: merge LoRA adapters and export model |
|   llamafactory-cli train -h: train models                          |
|   llamafactory-cli webchat -h: launch a chat interface in Web UI   |
|   llamafactory-cli webui: launch LlamaBoard                        |
|   llamafactory-cli version: show version info                      |
----------------------------------------------------------------------
```

**⚠️ Important**: ensure the correct kernel is selected

If the verification process fails, ensure the correct Jupyter kernel is selected for your notebook.
To change the kernel, follow these steps:

1. Go to the **Kernel** menu.
2. Select **Change Kernel**.
3. Select `Python 3 (ipykernel)` from the list.

**Important**: Failure to select the correct kernel can lead to unexpected issues when running the notebook.

### 6. Provide your Hugging Face token

You'll require a Hugging Face API token to access Llama-3.1. Generate your token at [Hugging Face Tokens](https://huggingface.co/settings/tokens) and request access for [Llama-3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct). Tokens typically start with "hf_". 

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

This section covers the process of setting up and running fine-tuning for the Llama-3 model using Llama-Factory. The following steps describe how to set up GPUs, import the required libraries, configure the model and training parameters, and run the fine-tuning process.

**⚠️ Important**: ensure the correct kernel is selected

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

### Download the model

Run the following command to download the weights to your local machine. This also downloads the tokenizer model and a responsible use guide.

To download Llama-3.1-8B-Instruct, use the following command:

```python
!huggingface-cli download --resume-download meta-llama/Llama-3.1-8B-Instruct
```

### Run the fine-tuning recipes

Now use the fine-tuning recipe `llama3_full_sft.yaml` for distributed training. This recipe predefines the [dataset identity](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/identity.json) and [alpaca_en_demo](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/alpaca_en_demo.json) JSON files.

**Note**: These datasets are only for evaluation purposes. You might need to change them to the actual ones.

The following command fine-tunes the Llama-3.1 8B model using the available GPUs.
>The command below saves a checkpoint every 500 steps. Before running the command, configure the appropriate `save_only_model` and `save_steps` values in the YAML file as required.

```python
!cd LLaMA-Factory/ && FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/llama3_full_sft.yaml model_name_or_path=meta-llama/Llama-3.1-8B-Instruct
```

### Monitoring GPU memory

To monitor GPU memory during training, run the following command in the terminal.

>**watch -n0 rocm-smi**

**Note**: For ROCm 6.5 and later, use the `amd-smi` command instead.

This command displays memory usage and other GPU metrics to ensure your hardware resources are being optimally used.

### Conclusion

LLaMA-Factory + DeepSpeed + ROCm provides a simple and efficient workflow for fine-tuning LLMs on AMD GPUs.
