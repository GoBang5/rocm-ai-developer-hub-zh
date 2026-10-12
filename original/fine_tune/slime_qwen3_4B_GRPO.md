> [Upstream source](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/fine_tune/slime_qwen3_4B_GRPO.ipynb) · Snapshot: `0071bfd4bd3bd9525405c09a6ae69c983b09312a`

# RL training with slime on AMD GPUs

**Author**: Wei Cai  
**Knowledge level**: Intermediate

Modern large language models don't stop improving after pretraining. To be useful, aligned, and robust for real-world tasks, they must learn from feedback. That's where Reinforcement Learning (RL) comes in.

This tutorial walks you through a real, production-style RL training pipeline for the [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) large language model, running entirely on AMD GPUs with ROCm. The workflow is powered by [slime](https://github.com/THUDM/slime), an SGLang-native post-training framework built specifically for RL scaling at LLM scale.

## Why use slime?

Training LLMs with RL is challenging for two key reasons:

- **Rollout generation is expensive**:
   You need a fast inference engine to sample large volumes of model responses.

- **Policy optimization is heavy**:
   You need a highly optimized training stack that scales across GPUs.

[slime](https://github.com/THUDM/slime) addresses both challenges by cleanly separating and efficiently connecting these two components:

- **SGLang**: For high-throughput rollout generation
- **Megatron-LM**: For distributed policy training

Together, they form a scalable, modular RL system that aligns with modern LLM workloads.

## What is GRPO?

This tutorial uses GRPO (Group Relative Policy Optimization), a modern RL algorithm designed for scalable LLM training.

Traditional RL methods often rely on a single reference baseline, also known as a critic model, which can be:

- Expensive to train
- Hard to stabilize
- Sensitive to reward noise

GRPO takes a different approach. Instead of evaluating a response in isolation, GRPO:

- Samples multiple responses for the same prompt
- Groups them together
- Computes relative advantages within the group

### Why this matters for LLMs

GRPO offers several practical advantages:
- No separate value network required
- More stable training signals
- Better scaling behavior for large batch rollouts
- Naturally fits server-based rollout generation (SGLang)

This makes GRPO especially well-suited for:
- Instruction tuning
- Reasoning improvement
- Preference-based optimization
- Large-scale RL on multi-GPU systems

## What you'll learn in this notebook

By the end of this tutorial, you'll be able to:

- Set up a ROCm-enabled Docker environment for slime on AMD GPUs
- Configure GRPO for Qwen3-4B, including rollout and reward settings
- Run an end-to-end RL training loop, combining:
  - SGLang for generation
  - Megatron-LM for optimization
- Understand the system-level design choices behind scalable LLM RL training

Whether you're experimenting with post-training research or building production-grade RL pipelines, this notebook is designed to give you both working code and clear mental models. Let's get started.

## Prerequisites

This tutorial was developed and tested using the following setup. 

### Operating system

* **Ubuntu 22.04**: Ensure your system is running Ubuntu 22.04.

### Hardware

* **AMD Instinct™ GPUs**: This tutorial was tested on a full node of eight AMD Instinct MI300X GPUs. Ensure you are using AMD Instinct GPUs or compatible hardware with ROCm support and that your system meets the [official requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

### Software

* **ROCm 7.0.0**: Install and verify ROCm by following the [ROCm install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html). After installation, confirm your setup using:

    ``` bash
    amd-smi
    ```

    This command lists your AMD GPUs with relevant details.

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
docker pull rlsys/slime:latest
```

### 2. Launch the Docker container

Launch the Docker container and map the necessary directories. 

``` bash
docker run -it \
  --device /dev/dri \
  --device /dev/kfd \
  -p 8265:8265 \
  --group-add video \
  --network host --ipc host \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --privileged \
  -v $HOME/.ssh:/root/.ssh \
  -v $HOME:$HOME \
  -w /workspace/notebooks \
  --shm-size 128G \
  --name slime \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  rlsys/slime:latest \
  /bin/bash
```

**Note**: If you need to return to the `slime` container after exiting it, use these commands:

``` bash
docker start slime
docker exec -it slime bash
```

**Note**: Ensure the notebook file is either copied to the `/workspace` directory or uploaded into the Jupyter Notebook environment after it starts. Save the token or URL provided in the terminal output to access the notebook from your web browser. You can download this notebook from the [AI Developer Hub GitHub repository](https://github.com/ROCm/gpuaidev).


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
Before starting RL training, you need to install slime — the core framework that connects
SGLang-based rollout generation with Megatron-LM-based policy optimization.

Since slime is under active development, ongoing commits could introduce behavior changes
that affect rollout semantics, reward computation, or training stability.
To ensure this tutorial is reproducible and stable, pin the installation to a known
working commit.

Run the following commands inside the ROCm-enabled Docker container:

```python
!git clone https://github.com/THUDM/slime.git
%cd slime
# Note -- You can run the latest upstream version. If you want a stable version, check out the following commit ID
!git checkout 0934a0e
# Install the package
!pip install -e .
```

Before continuing, confirm that slime is correctly installed and visible to Python:

```python
# Verify the installation and version of the required libraries
!pip list | grep slime
```

If `slime` appears in the output, your environment is ready for the next step.

## Run GRPO training

This section walks you through the end-to-end process of setting up and running
Group Relative Policy Optimization (GRPO) training for Qwen3-4B.

At a high level, GRPO training requires:
- A base pretrained model (the policy you want to improve)
- A training dataset to use to generate rollouts and compute rewards
- A held-out evaluation dataset to track generalization during training

### 1. Download the model and datasets

First, download the base Qwen3-4B model, which serves as the initial policy
for RL fine-tuning.

For training, you will use `dapo-math-17k`, a dataset designed to evaluate
step-by-step mathematical reasoning. This is a task for which relative comparisons
between multiple model outputs are especially effective.

For evaluation, use `aime-2024`, which provides a clean benchmark to
monitor reasoning performance without leaking training data.

Run the following commands to download all required artifacts. First, download the base Qwen3-4B model checkpoint from Hugging Face:

```python
!hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B
```

Next, download the training dataset (`dapo-math-17k`) for mathematical reasoning tasks:

```python
!hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k
```

Then download the AIME 2024 evaluation dataset for benchmarking:

```python
!hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /root/aime-2024
```

### 2. Convert the checkpoint format

Before you can start GRPO training, convert the pretrained Hugging Face checkpoint into the Megatron Core distributed format.

This conversion is required because slime uses Megatron-LM for training. Megatron-LM expects model weights to be laid out according to the target parallelization strategy (that is, tensor parallelism and pipeline parallelism). Hugging Face checkpoints, by contrast, store weights in a framework-agnostic, single-process format.

This is a one-time preprocessing step for a given model and parallel configuration. You don't need to repeat it for every training run, as long
as the parallelism settings remain unchanged.

Run the following commands to perform the conversion:

```python
%%bash
# Navigate to the slime repository
cd /workspace/notebooks/slime

# Load model configuration arguments
source scripts/models/qwen3-4B.sh

# Locate megatron core installation path
MEGATRON_LM_PATH=$(pip list | grep megatron-core | awk '{print $NF}')

# Run conversion tool
PYTHONPATH=${MEGATRON_LM_PATH} python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --no-gradient-accumulation-fusion \
    --hf-checkpoint /root/Qwen3-4B \
    --save /root/Qwen3-4B_torch_dist
```

### 3. Launch GRPO training

This step prepares and launches the GRPO training runtime.

Because slime training scripts are designed to run in standalone environments, they might include safeguards (such as `pkill -9 python`) that are unsafe when run inside a Jupyter notebook. Additionally, certain offloading behaviors can cause instability on AMD GPUs in interactive environments.

To ensure a stable Jupyter-based workflow, the next cell performs the following actions:

1. Prevents the training script from terminating the Jupyter kernel.
2. Inserts required `--no-offload` flags for both training and rollout.
3. Reduces the rollout count to make early performance improvements easier to observe.

**Note**: You can adjust `--num-rollout` based on your dataset size and training goals. A larger value for `--num-rollout` results in more rollouts per iteration, effectively increasing the training epoch and improving convergence at the cost of a longer runtime.

Run the following commands as a one-time setup task to patch the training script.

```python
%%bash
# Navigate to the slime repository
cd /workspace/notebooks/slime

SCRIPT=scripts/run-qwen3-4B-amd.sh

echo "Patching $SCRIPT ..."

# 1. Comment out `pkill -9 python` only if it is not already commented
if grep -qE '^[[:space:]]*pkill -9 python' "$SCRIPT"; then
  echo " - Commenting out pkill -9 python"
  sed -i 's/^[[:space:]]*pkill -9 python/# pkill -9 python/' "$SCRIPT"
else
  echo " - pkill already commented or not present"
fi

# 2. Inject no-offload flags only if they are not already present
if ! grep -q -- '--no-offload-train' "$SCRIPT"; then
  echo " - Injecting --no-offload flags after --colocate"
  sed -i '/--colocate/a \   --no-offload-train \\\n   --no-offload-rollout \\' "$SCRIPT"
else
  echo " - no-offload flags already present"
fi

sed -i 's/--num-rollout[[:space:]]\+[0-9]\+/--num-rollout 200/' "$SCRIPT"

echo "Patch completed."
```

After the script is patched, start the GRPO training loop:

```python
%%bash
# Navigate to the slime repository
cd /workspace/notebooks/slime

# Launch the training script with environment variables set
SLIME_DIR=/root \
MODEL_DIR=/root \
DATA_DIR=/root \
bash scripts/run-qwen3-4B-amd.sh
```

### 4. Understanding the training script

The `run-qwen3-4B-amd.sh` script contains all the configuration for GRPO training. It's organized into several parameter groups that control different aspects of the training pipeline.

Below is a breakdown of the script's key components. Each section corresponds to a specific aspect of the training workflow.

#### Model configuration

```bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-4B.sh"
```

This loads the model architecture settings from `scripts/models/qwen3-4B.sh`. These are Megatron-LM parameters that define the model structure.

**Important**: Ensure settings such as `--rotary-base` match your target model. Different models might use different rotary base values. If necessary, override this setting after sourcing the configuration using the following commands:

```bash
source "${SCRIPT_DIR}/models/qwen3-4B.sh"
MODEL_ARGS+=( --rotary-base 10000 )
```

#### Checkpoint configuration

```yaml
CKPT_ARGS=(
   # HF checkpoint required by SGLang; also used for tokenizer
   --hf-checkpoint ${MODEL_DIR}/Qwen3-4B
   # Reference model checkpoint
   --ref-load ${MODEL_DIR}/Qwen3-4B_torch_dist
   # Actor model load directory; if empty, loads from ref_load
   --load ${MODEL_DIR}/Qwen3-4B_slime/
   --save ${MODEL_DIR}/Qwen3-4B_slime/
   --save-interval 20
)
```

This controls the location where models are loaded from, and saved to, during training.

#### Rollout configuration

```yaml
ROLLOUT_ARGS=(
   # Dataset configuration
   --prompt-data ${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   # Reward model
   --rm-type deepscaler
   
   # Rollout parameters
   --num-rollout 200
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 0.8
   
   # Training batch configuration
   --global-batch-size 256
   --balance-data
)
```

**Key parameters**:
- `--num-rollout`: Total number of rollouts for training
- `--n-samples-per-prompt`: Responses sampled per prompt (used for group-relative advantages in GRPO)
- `--rm-type`: Reward model type (slime supports multiple types and custom models using `--custom-rm-path`)

#### Evaluation configuration

The evaluation task inherits rollout settings but allows you to override specific parameters:

```yaml
EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime ${DATA_DIR}/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 0.7
)
```

#### Performance and parallelism

```yaml
PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)
```

**Key optimizations**:
- `--use-dynamic-batch-size`: Packs samples of varying lengths into micro batches up to the token limit
- `--max-tokens-per-gpu`: Hard limit of tokens per GPU

**Note**: slime guarantees strict per-token loss calculation even with dynamic packing.

#### GRPO algorithm parameters

```yaml
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)
```

#### Optimizer configuration

```yaml
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)
```

#### SGLang configuration

```yaml
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2  # SGLang tensor parallelism
   --sglang-mem-fraction-static 0.7
)
```

Arguments prefixed with `--sglang-` are forwarded directly to the SGLang engine.

### 5. Convert from Megatron format to Hugging Face format for post-training inference

After RL training with slime, the model checkpoints are saved in the Megatron-LM distributed format, which is not directly usable for standard inference frameworks. To run inference with Hugging Face Transformers or SGLang, you must convert these checkpoints back to Hugging Face (HF) format.

To convert the trained Megatron checkpoint (from a specific training iteration) back into Hugging Face format, use these commands:

```python
%%bash
# Navigate to the slime repository
cd /workspace/notebooks/slime

# Load model configuration arguments
source scripts/models/qwen3-4B.sh

# Locate megatron-core installation path
MEGATRON_LM_PATH=$(pip list | grep megatron-core | awk '{print $NF}')

PYTHONPATH=${MEGATRON_LM_PATH} python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --no-gradient-accumulation-fusion \
    --hf-checkpoint /root/Qwen3-4B \
    --save /root/Qwen3-4B_torch_dist

PYTHONPATH=${MEGATRON_LM_PATH} python tools/convert_torch_dist_to_hf.py \
  --input-dir /root/Qwen3-4B_slime/iter_0000199 \
  --output-dir /root/Qwen3-4B_slime_hf-iter_0000199 \
  --origin-hf-dir /root/Qwen3-4B
```

After conversion, the data in `/root/Qwen3-4B_slime_hf_iter_0000199` is in the standard Hugging Face format, ready for [transformers inference](https://huggingface.co/Qwen/Qwen3-4B), [SGLang serving](https://docs.sglang.io/basic_usage/send_request.html), and further evaluation or fine-tuning.

## Summary

Congratulations! By working through this GRPO training tutorial with slime, you learned how to train large language models using reinforcement learning on AMD GPUs.

Here are the key takeaways:

- **Environment setup**: ROCm-enabled Docker containers with slime provide a complete RL training environment.
- **Checkpoint management**: Converting between Hugging Face and Megatron formats enables seamless integration across frameworks.
- **GRPO training**: Group-relative advantages provide stable RL training without requiring a separate value network.
- **Scalable architecture**: SGLang rollout generation and Megatron-LM policy optimization work together efficiently.

## Next steps

1. **Experiment with different datasets**: Apply GRPO to other reasoning or instruction-following datasets.
2. **Tune hyperparameters**: Adjust the learning rate, KL coefficients, or sampling strategies for your specific use case.
3. **Scale to larger models**: Use the same workflow with larger Qwen models or other LLM architectures.
4. **Evaluate trained models**: Test your fine-tuned models on downstream tasks to measure improvement.

## Additional resources

- [slime GitHub repository](https://github.com/THUDM/slime)
- [SGLang documentation](https://docs.sglang.io/)
- [Megatron-LM documentation](https://github.com/NVIDIA/Megatron-LM)
- [ROCm documentation](https://rocm.docs.amd.com/en/latest/index.html)
- [AMD AI Developer Hub](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/)

```python

```
