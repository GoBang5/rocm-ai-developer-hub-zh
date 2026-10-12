> [Upstream source](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/pretrain/training_with_primus.ipynb) · Snapshot: `0071bfd4bd3bd9525405c09a6ae69c983b09312a`

# Training a model with Primus

**Author**: Wei Cai  
**Knowledge level**: Intermediate

This tutorial demonstrates how to pretrain the [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) large language model (LLM) using ROCm on AMD GPUs by leveraging Primus. Primus is a unified and flexible training framework for AMD Instinct™ GPUs that's designed to support multiple training engine backends, including Megatron, to deliver scalable, high-performance model training. Performance acceleration is powered by [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo) and the ROCm libraries.

Primus with Megatron is designed to replace the [ROCm Megatron-LM training](https://rocm.docs.amd.com/en/docs-7.2.4/how-to/rocm-for-ai/training/benchmark-docker/megatron-lm.html) workflow. To learn how to migrate workloads from Megatron-LM to Primus with Megatron, see [Migrating workloads to Primus (Megatron backend) from Megatron-LM](https://rocm.docs.amd.com/en/docs-7.2.4/how-to/rocm-for-ai/training/benchmark-docker/previous-versions/megatron-lm-primus-migration-guide.html).

By following this tutorial, you'll learn how to set up and run training with Primus, take advantage of its performance optimizations, and gain practical experience in using Primus to streamline model training workflows.

## Prerequisites

This tutorial was developed and tested using the following setup. 

### Operating system

* **Ubuntu 22.04**: Ensure your system is running Ubuntu 22.04.

### Hardware

* **AMD Instinct™ GPUs**: This tutorial was tested on a full node of AMD Instinct MI300X GPUs (eight MI300X GPUs). Ensure you are using AMD Instinct GPUs or compatible hardware with ROCm support and that your system meets the [official requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

### Software

* **ROCm 7.0.0**: Install and verify ROCm by following the [ROCm install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html). After installation, confirm your setup using:

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

## System validation

Before running AI workloads, it's important to ensure that your AMD hardware is configured correctly and performing optimally.

Generally, application performance can benefit from disabling NUMA (Non-Uniform Memory Access) auto-balancing. However, this setting might be detrimental to performance with certain types of workloads.

Run this command to verify the current NUMA settings:

``` bash
cat /proc/sys/kernel/numa_balancing
``` 

An output of `0` indicates NUMA auto-balancing is disabled. If there is no output or the output is `1`, run the following command to disable NUMA auto-balancing.

``` bash
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
``` 

For more information, see [Disable NUMA auto-balancing](https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/system-optimization/mi300x.html#disable-numa-auto-balancing).

## Set up the environment

Follow these steps to prepare the training environment.

### 1. Pull the Docker image

Ensure your system meets the [System Requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

Pull the Docker image required for this tutorial:

``` bash
docker pull rocm/primus:v25.9_gfx942
```

### 2. Launch the Docker container

Launch the Docker container and map the necessary directories. 

``` bash
docker run -it \
    --device /dev/dri \
    --device /dev/kfd \
    --device /dev/infiniband \
    --network host --ipc host \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    --shm-size 128G \
    --name primus_training_env \
    rocm/primus:v25.9_gfx942
```

**Note**: If you need to return to the `primus_training_env` container after exiting it, use these commands:

``` bash
docker start primus_training_env
docker exec -it primus_training_env bash
```

**Note**: Ensure the notebook file is either copied to `/workspace` directory or uploaded into the Jupyter Notebook environment after it starts. Save the token or URL provided in the terminal output to access the notebook from your web browser. You can download this notebook from the [AI Developer Hub GitHub repository](https://github.com/ROCm/gpuaidev).


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

## Edit the training configuration

Primus defines a training configuration in YAML for each model in `examples/megatron/configs`.

For example, to update training parameters for Qwen2.5-7B, update `examples/megatron/configs/primus_qwen2.5_7B-pretrain.yaml`. The YAML training configuration files for other models follow this naming convention.

### Dataset options

You can use either mock data or real data for training.

- Mock data can be useful for testing and validation. Use the `mock_data` field to toggle between mock and real data. The default value is `true` (for enabled).

  ``` yaml
  mock_data: true
  ```

- If you’re using a real dataset, update the `train_data_path` field to point to your dataset location.

  ``` yaml
  mock_data: false
  train_data_path: /path/to/your/dataset
  ```

Ensure that the files are accessible inside the Docker container.

```python
mock_data = True   # Set to False if you want to use a real dataset
train_data_path = "/path/to/your/dataset"  # Only used if mock_data=False

# Conditional logic
if mock_data:
    print("✅ Using mock dataset for testing and validation.")
    # Insert mock dataset loading logic here
else:
    print(f"✅ Using real dataset from: {train_data_path}")
    # Example: verify the dataset path exists
    import os
    if os.path.exists(train_data_path):
        print("📂 Dataset found and accessible.")
        # Insert real dataset loading logic here
    else:
        print("❌ Dataset path not found! Please check your configuration.")
```

## Run training
Use the following example commands to set up the environment, configure [key options](https://rocm.docs.amd.com/en/docs-7.2.4/how-to/rocm-for-ai/training/benchmark-docker/previous-versions/primus-megatron-v25.8.html?model=primus_pyt_megatron_lm_train_qwen2.5-7b), and run training on AMD Instinct GPUs using Primus with the Megatron backend.

### Single node training
To run training on a single node, navigate to `/workspace/Primus` and use the setup commands in the following section:

```python
%cd /workspace/Primus
```

#### 1. Set environment variables

Run the following commands to set up and confirm the environment variables.

```python
# Install dependencies
!pip install -r requirements.txt

# Set environment variables for current process
import os
os.environ["HSA_NO_SCRATCH_RECLAIM"] = "1"
os.environ["NVTE_CK_USES_BWD_V3"] = "1"

# (Optional) Print to confirm they are set
print("HSA_NO_SCRATCH_RECLAIM =", os.environ.get("HSA_NO_SCRATCH_RECLAIM"))
print("NVTE_CK_USES_BWD_V3 =", os.environ.get("NVTE_CK_USES_BWD_V3"))
```

After setup is complete, run the appropriate training commands. The following run commands are tailored to Qwen2.5-7B. See [Supported models](https://rocm.docs.amd.com/projects/primus/en/latest/06-developer-guide/model-support-matrix.html) to switch to another available model.

#### 2. Show the training configuration

````python
# Display the training config YAML file
import yaml
from IPython.display import Markdown, display

with open("examples/megatron/configs/qwen2.5_7B-pretrain.yaml", "r") as f:
    train_config = yaml.safe_load(f)

yaml_text = yaml.dump(
    train_config,
    sort_keys=False,
    default_flow_style=False
)

display(Markdown(f"```yaml\n{yaml_text}\n```"))
````

You can modify the parameters directly in the YAML file or override them using CLI arguments.

#### 3. Launch training

To run training on a single node for Qwen2.5-7B, use the following command:

```python
import subprocess

# Choose training mode: "bf16" or "fp8"
mode = "bf16"  # change to "fp8" if you want FP8 training

if mode == "bf16":
    cmd = """
    EXP=examples/megatron/configs/qwen2.5_7B-pretrain.yaml \
    bash examples/run_pretrain.sh \
        --train_iters 50
    """
elif mode == "fp8":
    cmd = """
    EXP=examples/megatron/configs/qwen2.5_7B-pretrain.yaml \
    bash examples/run_pretrain.sh \
        --train_iters 50 \
        --fp8 hybrid
    """
else:
    raise ValueError("Unsupported mode. Use 'bf16' or 'fp8'.")

print("Running command:\n", cmd)
subprocess.run(cmd, shell=True, check=True)
```

### Key training options

The following key options apply to the training commands shown above:

**fp8**

* `hybrid` enables `FP8` GEMMs.

**use_torch_fsdp2**

* `use_torch_fsdp2: 1` enables torch FSDP-v2.
* If FSDP is enabled, set `use_distributed_optimizer` and `overlap_param_gather` to `false`.

**profile**

* To enable PyTorch profiling, set these parameters:

  ``` yaml
  profile: true
  use_pytorch_profiler: true
  profile_step_end: 7
  profile_step_start: 6
  ```

**train_iters**

* The total number of iterations (default is `50`).

**mock_data**

* `True` by default.

**micro_batch_size**

* Micro batch size.

**global_batch_size**

* Global batch size.

**recompute_granularity**

* For activation checkpointing.

**num_layers**

* For using a reduced number of layers, for example, with proxy models.

## Further reading
- For an introduction to Primus, see [Primus: A Lightweight, Unified Training Framework for Large Models on AMD GPUs](https://rocm.blogs.amd.com/software-tools-optimization/primus/README.html).
- To learn how to set up and run training with Primus in combination with PyTorch, see [Training a model with Primus and Pytorch TorchTitan](https://rocm.docs.amd.com/projects/primus/en/latest/02-user-guide/torchtitan-training.html).
- To learn more about system settings and management practices to configure your system for AMD Instinct MI300X series GPUs, see [AMD Instinct MI300X system optimization](https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/system-optimization/mi300x.html).
- For a list of other ready-made Docker images for AI with ROCm, see [AMD Infinity Hub](https://www.amd.com/en/developer/resources/infinity-hub.html#f-amd_hub_category=AI%20%26%20ML%20Models).
