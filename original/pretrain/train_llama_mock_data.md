> [Upstream source](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/pretrain/train_llama_mock_data.ipynb) · Snapshot: `0071bfd4bd3bd9525405c09a6ae69c983b09312a`

# Training Llama-3.1 8B with Megatron-LM 

**Authors**: Shekhar Pandey and Liz Li  
**Knowledge level**: Intermediate

This tutorial demonstrates how to train the Llama-3.1 model using *mock data*. The Llama-3.1 8B model is a popular open-source large language model (LLM) designed to handle a wide range of natural language processing tasks efficiently. Learn more about the Llama models at [Llama's website](https://www.llama.com/).

This tutorial uses mock data to provide a quick and lightweight demonstration of the training workflow, enabling you to verify that your environment is correctly configured and functional. Mock data is a useful way to validate the training pipeline without requiring large datasets.

The training process leverages the Megatron-LM framework, a specialized framework for pretraining and fine-tuning large-scale language models. For more information about Megatron-LM, see their [GitHub repository](https://github.com/NVIDIA/Megatron-LM). All steps are executed within a Docker container, which provides a ready-to-use environment with all necessary dependencies.

This tutorial builds on the setup completed in the [Pretraining with Megatron-LM tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/setup_tutorial.html).

## Prerequisites

This tutorial was developed and tested using the following setup. 

### Operating system

* **Ubuntu 22.04**: Ensure your system is running Ubuntu version 22.04.

### Hardware

* **AMD Instinct™ GPUs**: This tutorial was tested on an AMD Instinct MI300X GPU. Ensure you are using an AMD Instinct GPU or compatible hardware with ROCm support and that your system meets the [official requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

### Software

* **ROCm 6.2**: Install and verify ROCm by following the [ROCm install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html). After installation, confirm your setup using:

    ``` bash
    amd-smi
    ```

    This command lists your AMD GPUs with relevant details.
    
    **Note**: For ROCm 6.4 and earlier, use the `rocm-smi` command instead.

* **Docker**: Ensure Docker is installed and configured correctly. Follow the Docker installation guide for your operating system.

   **Note**: Ensure the Docker permissions are correctly configured. To configure permissions to allow non-root access, run the following commands:

   ```bash
   sudo usermod -aG docker $USER
   newgrp docker
   ```

   Verify Docker is working correctly:

   ```bash
   docker run hello-world
   ```

### Hugging Face API access

* Obtain an API token from [Hugging Face](https://huggingface.co) for downloading models.
* Ensure the Hugging Face API token has the necessary permissions and approval to access [Meta's Llama checkpoints](https://huggingface.co/meta-llama/Llama-3.1-8B).

## Prepare the training environment

After your system meets the prerequisites, follow these steps to set up the training environment.

### 1. Pull the Docker image

Run the following command in your terminal to pull the prebuilt Docker image. The Docker image provides all necessary dependencies, including PyTorch, PyTorch Lightning, ROCm libraries, and Megatron-LM utilities.

``` bash
docker pull rocm/megatron-lm:24.12-dev
```

### 2. Launch the Docker container

Run the following command in your terminal to launch the Docker container with the appropriate configuration:

```bash
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

**Note**: This command mounts the current directory to the `/workspace` directory in the container. Ensure the notebook file is either copied to this directory before running the command or uploaded into the Jupyter Notebook environment after it starts. Save the token or URL provided in the terminal output to access the notebook from your web browser. You can download this notebook from the [AI Developer Hub GitHub repository](https://github.com/ROCm/gpuaidev).

### 3. Install Jupyter and start the server

Inside the Docker container, install Jupyter using the following command:

```bash
pip install jupyter
```

Start the Jupyter server:

```bash
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```
**Note**: Ensure port `8888` is not already in use on your system before running the above command. If it is, specify a different port by replacing `--port=8888` with another port number, for example, `--port=8890`.

### 4. Access the Jupyter notebook

There are two ways to load your Jupyter notebook:

#### Mount the notebook with Docker

* **Ensure the correct mounting**: When launching the container, use the `-v /path/to/notebooks:/workspace/notebooks` option to mount your notebook directory.
* **Locate the notebook in the container**: Inside the Jupyter interface, navigate to `/workspace/notebooks` to find your files and begin working on them.

#### Upload the notebook through the browser

* **Skip the `-v` option when launching Docker**: If you prefer not to mount your notebook directory, omit the `-v /path/to/notebooks:/workspace/notebooks` option.
* **Upload manually**: After accessing the Jupyter interface in your browser:
  1. Click the **Upload** button in the top-right corner.
  2. Select your notebook file from your host machine and upload it to the container.
* **Start working**: After the notebook is uploaded, click the notebook filename to open it and begin working.

### 5. Clone the Megatron-LM repository

Run the following commands inside the Docker container to clone the Megatron-LM repository and navigate to the validated commit:

```python
# Clone the Megatron-LM repository and navigate to the validated commit
!git clone https://github.com/ROCm/Megatron-LM && cd Megatron-LM && git checkout bb93ccbfeae6363c67b361a97a27c74ab86e7e92
```

### 6. Provide your Hugging Face token

A Hugging Face token can be generated by signing into your account at [Hugging Face Tokens](https://huggingface.co/settings/tokens).

You'll require a Hugging Face API token to access Llama-3.1 8B. Generate your token at Hugging Face Tokens and request access for Llama-3.1 8B. Tokens typically start with "hf_".

Run the following interactive block in your Jupyter notebook to set up the token:

**Note**: Uncheck the "Add token as Git credential" option.

```python
from huggingface_hub import notebook_login, HfApi

# Prompt the user to log in
notebook_login()
```

Verify that your token was accepted correctly:

```python
try:
    api = HfApi()
    user_info = api.whoami()
    print(f"Token validated successfully! Logged in as: {user_info['name']}")
except Exception as e:
    print(f"Token validation failed. Error: {e}")
```

## Run the training script

This section describes how to run the training script, with an explanation of the key parameters.

### Single-node training overview

The training process involves running a pre-configured script that initializes and executes the training of the Llama-3.1 model. The script leverages the Megatron-LM framework and mock data to simulate a full training pipeline. This approach ensures your environment is configured correctly and is functional for real-world use cases.

Before running the script, ensure all environment variables are set correctly.

### Key parameters for training:

* **Batch size (`BS`)**: Set this to `64` for optimal GPU usage.
* **Sequence length (`SEQ_LENGTH`)**: Input sequence length, set to `4096`.
* **Tensor parallelism (`TP`)**: Set this to `8` for efficient parallelism.
* **Precision (`TE_FP8`)**: Set this to `0` for `BF16` precision.

### Run the training script

Use the following command to train the model on a single node:

```python
!cd Megatron-LM && TEE_OUTPUT=1 MBS=2 BS=64 TP=8 TE_FP8=0 SEQ_LENGTH=4096  \
TOKENIZER_MODEL='meta-llama/Llama-3.1-8B' MODEL_SIZE='8' \
bash examples/llama/train_llama3.sh
```

### Additional details about the command

This command configures the training process with the following parameters:

* **`TEE_OUTPUT=1`**: Enables logging output to the console.
* **`MBS=2`**: Micro-batch size per GPU.
* **`BS=64`**: Total batch size across all GPUs.
* **`TP=8`**: Tensor parallelism for distributing the model across GPUs.
* **`TE_FP8=0`**: Sets the precision to `BF16` for training.
* **`SEQ_LENGTH=4096`**: Maximum input sequence length.

The training script does the following:
* Uses mock data as input.
* Trains the Llama-3.1 8B model with the specified configurations.

You can customize these parameters based on your hardware and desired configurations by modifying the command details.

## Monitor the training progress

Monitor the output logs during the training process for the following developments:

* **Iteration progress**: The number of completed iterations.
* **Loss values**: This indicates the model's learning progress. Lower values suggest better learning.
* **GPU utilization**: Ensures the optimal usage of your hardware resources.

Logs are printed to the console and saved to a log file within the directory specified by the script.

## Key notes

* Mock data is for validation only. To use a different dataset, see the [Pretraining with Megatron-LM tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/setup_tutorial.html).
* Tune the hyperparameters based on your hardware. The hyperparameter configuration in this tutorial is based on one node of 8x MI300x GPUs.
* This example illustrates how to run a training task on a single node. For multi-node training instructions, see the [Pretraining with Megatron-LM tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/setup_tutorial.html).
* Verify the logs for correctness.
