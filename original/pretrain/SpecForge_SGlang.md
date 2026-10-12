> [Upstream source](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/pretrain/SpecForge_SGlang.ipynb) · Snapshot: `0071bfd4bd3bd9525405c09a6ae69c983b09312a`

# SpecForge: Training speculative decoding models with SGLang

**Author**: Ning Zhang  
**Knowledge level**: Intermediate

Speculative decoding is a powerful technique to improve inter-token latency in memory-bound LLM inference, which needs to run two models in parallel. These are the *target model* (the main LLM model for the AI application) and the *small draft model* (a smaller, lightweight LLM that runs alongside it to help speed up inference for the main LLM). The small draft model is important in speculative decoding because its ability to correctly predict the tokens for target model acceptance is the key factor in whether the deployment will be successful or not.

[SpecForge](https://github.com/sgl-project/SpecForge), a purpose-built ecosystem for training draft models that integrate natively with SGLang, is an open-source application that is listed as a flagship project by the Large Model Systems Organization (LMSYS). This tutorial demonstrates how to run SpecForge draft model training and SGLang speculative decoding inference with the trained draft model on a system with an AMD Instinct™ MI300X GPU.  

## Prerequisites
This tutorial was developed and tested using the following setup.

### Operating system
* **Ubuntu 22.04/24.04**: Ensure your system is running Ubuntu 22.04 or 24.04.

### Hardware
* **AMD Instinct GPUs**: This tutorial was tested on an AMD Instinct MI300X GPU node with eight GPUs. Ensure you are using a node with eight AMD Instinct GPUs or compatible hardware with ROCm support and that your system meets [the official requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

### Software
* **ROCm 6.3+**:  This tutorial requires ROCm 6.3 or later. Install and verify ROCm by following the [ROCm install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html). After installation, confirm your setup using the `amd-smi` command. AMD also provides prebuilt ROCm Docker images, for example, [ROCm PyTorch](https://hub.docker.com/r/rocm/pytorch), [ROCm Ubuntu 22.04](https://hub.docker.com/r/rocm/dev-ubuntu-22.04), and [ROCm Ubuntu 24.04](https://hub.docker.com/r/rocm/dev-ubuntu-24.04). You can use these prebuilt Docker images to reduce the effort of setting up a ROCm environment.

  **Note**: For ROCm 6.4 and earlier, use the `rocm-smi` command instead of `amd-smi`.

* **Docker**: Ensure Docker is installed and configured correctly. Follow the Docker installation guide for your operating system.

   **Note**: Ensure the Docker permissions are correctly configured. To configure permissions to allow non-root access, run the following commands:

   ``` bash
   sudo usermod -aG docker $USER
   newgrp docker
   ```

   Verify Docker is working correctly with:

   ``` bash
   docker run hello-world
   ```

### Hugging Face API access

* Obtain an API token from [Hugging Face](https://huggingface.co) for downloading models.
* Ensure the Hugging Face API token has the necessary permissions and approval to access the [Meta Llama checkpoints](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).

## Set up the SpecForge environment 

In this tutorial, you will work on the prebuilt ROCm PyTorch image as an example. You can also try the other ROCm images.

### Step 1: Launch the Docker image 

Launch the Docker container. Replace `/path/to/SpecForge_Project` with the full path to the directory on your host machine where the SpecForge code and model files are stored. This tutorial uses the [SGLang ROCm Docker images](https://hub.docker.com/r/lmsysorg/sglang/tags?name=rocm) to demonstrate SpecForge draft model training and SGLang speculative decoding inference. For better model training performance, you can also try the [ROCm PyTorch Training Docker image](https://hub.docker.com/r/rocm/pytorch-training/).

``` bash
docker run -it --rm \
  --network=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size 32G \
  -v /path/to/SpecForge_Project:/SpecForge \
  -w /SpecForge/ \
  lmsysorg/sglang:v0.4.10.post2-rocm630-mi30x
```
**Note**: This command mounts the current directory to the `/SpecForge` directory in the container. Ensure the notebook file is either copied to this directory before running the Docker command or uploaded into the Jupyter Notebook environment after it starts. Save the token or URL provided in the terminal output to access the notebook from your web browser. You can download this notebook from the [AI Developer Hub GitHub repository](https://github.com/ROCm/gpuaidev).

### Step 2: Install and launch Jupyter

Inside the Docker container, install Jupyter using the following command:

``` bash
pip install jupyter
```

Start the Jupyter server:

``` bash
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**Note**: Ensure port `8888` is not already in use on your system before running the above command. If it is, you can specify a different port by replacing `--port=8888` with another port number, for example, `--port=8890`.

### Step 3: Install SpecForge

Directly installing the [official SpecForge code](https://github.com/sgl-project/SpecForge) also installs some non-AMD related GPU libraries, which is not recommended. To avoid this issue, pick up the correct [commit](https://github.com/zhangnju/SpecForge) and make some modifications. Run the following commands inside the Jupyter notebook running within the Docker container:

```python
%%bash
git clone https://github.com/zhangnju/SpecForge.git
pip install -v ./SpecForge
```

### Step 4: Provide your Hugging Face token

You'll require a Hugging Face API token to access Llama-3. Generate your token at [Hugging Face Tokens](https://huggingface.co/settings/tokens) and request access for [Llama-3 8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct). Tokens typically start with "hf_". 

Run the following interactive block in your Jupyter notebook to set up the token:

**Note**: Uncheck the "Add token as Git credential" option.

```python
from huggingface_hub import notebook_login

# Prompt the user to log in
notebook_login()
```

```python
from huggingface_hub import HfApi

try:
    api = HfApi()
    user_info = api.whoami()
    print(f"Token validated successfully! Logged in as: {user_info['name']}")
except Exception as e:
    print(f"Token validation failed. Error: {e}")
```

## Speculative decoding draft model training

SpecForge is a framework for training speculative decoding models so you can smoothly port them over to the SGLang serving framework to speed up your inference. It offers two methods of training the draft model: online and offline training. Online training freezes the target model and training draft model at same time, which generates auxiliary hidden states on the fly and needs multiple GPUs to achieve better performance. Offline training generates and saves the hidden states using the target model first and then trains the draft model in a separate process. Offline training requires as little as one GPU because it only needs to accommodate the draft model, but it needs a huge amount of disk space. For example, the UltraChat and ShareGPT datasets would need 12TB of storage. Due to the disk size of the AMD Instinct MI300X system, this tutorial uses online training to demonstrate SpecForge.  

### Data preparation 

SpecForge provides a script to prepare the UltraChat (200k) and ShareGPT (120k) datasets for draft model training. The commands below process the datasets into `jsonl` (JSON Lines) files, which are the raw dataset for online training, and place them in the `cache/dataset/<dataset_name>` directory. To train the model using your own data, prepare the dataset in `jsonl` format. The schema should look like this:

```json
{
    "id": "xxxx",
    "conversations": [
        {
            "role": "user | assistant",
            "content": "The message content"
        }
    ],
}
```

This tutorial uses the ShareGPT and UltraChat datasets as examples.

```python
%%bash
# ultrachat
python SpecForge/scripts/prepare_data.py --dataset ultrachat

# sharegpt
python SpecForge/scripts/prepare_data.py --dataset sharegpt
```

### Draft model online training 

SpecForge provides sample scripts to train the draft model for Llama3/4, Qwen3, and other popular LLM models. This tutorial uses the Llama-3 8B model to demonstrate how to run online training using SpecForge. Because model training takes a long time, you will only run one epoch of training, which might not have the best accuracy. You can change the training options in the command below as required.

```python
%%bash
# Llama-3 8B draft model training using 8 GPUs 
torchrun \
    --standalone \
    --nproc_per_node 8 \
    SpecForge/scripts/train_eagle3_online.py \
    --target-model-path meta-llama/Meta-Llama-3-8B-Instruct \
    --draft-model-config SpecForge/configs/llama3-8B-eagle3.json \
    --train-data-path SpecForge/cache/dataset/sharegpt.jsonl \
    --output-dir outputs/llama3-8b-eagle3 \
    --num-epochs 1 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template llama3 \
    --cache-dir cache
```

After the training is done, you can find the trained draft model file in the `output-dir` directory.

## (Optional) SGLang speculative decoding inference 

This section shows how to test the trained draft model based on the SGLang framework. If you are already familiar with SGLang speculative decoding inference, you can skip this section.

### SGLang speculative decoding server

The SGLang inference server supports EAGLE speculative decoding through the following options:
1) `speculative-draft-model-path`: Specifies the draft model. This parameter is required.
2) `speculative-num-steps`: Specifies the depth of auto-regressive drafting. This variable increases the speculation range but risks rejection cascades. The default is `5`.
3) `speculative-eagle-topk`: Specifies the branching factor per step. This improves candidate diversity. It leads to a higher acceptance rate but also causes higher memory/compute consumption. The default is `4`.
4) `speculative-num-draft-tokens`: Specifies the maximum parallel verification capacity. It causes deeper tree evaluation but leads to higher GPU memory usage. The default is `8`. 

The settings for these options in the following SGLang server command are intended to demonstrate the speculative decoding functional test and are not about maximizing performance. You can find the best combinations of these parameters in [bench_speculative.py](https://github.com/sgl-project/sglang/blob/main/scripts/playground/bench_speculative.py).

**Note**: Run the commands in this section from a terminal, not from the notebook code cells. In JupyterLab, open a terminal using **Launcher** → **Terminal** (or **File** → **New** → **Terminal**).

```python
%%bash
python3 -m sglang.launch_server --model meta-llama/Meta-Llama-3-8B-Instruct \
         --speculative-algorithm EAGLE3 --speculative-draft-model-path ./outputs/llama3-8b-eagle3/epoch_0 \
         --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
         --mem-fraction-static 0.75 --cuda-graph-max-bs 2 --tp 8 --context-length 8192 --trust-remote-code \
         --host 0.0.0.0 --port 30000 --dtype bfloat16 --attention-backend triton
```

### SGLang speculative decoding benchmarking 

SpecForge provides multiple benchmark scripts for speculative decoding. After launching the SGLang inference server, as shown above, you can run these benchmarking scripts to test speculative decoding performance with different test datasets.

```python
%%bash
# GSM8K
python SpecForge/benchmarks/run_gsm8k.py

# MATH-500
python SpecForge/benchmarks/run_math500.py

# MTBench
python SpecForge/benchmarks/run_mtbench.py

# HumanEval
python SpecForge/benchmarks/run_humaneval.py
```

## Summary 
Speculative decoding has emerged as a breakthrough for accelerating LLM inference. SpecForge is designed for speculative decoding draft model training and is tightly integrated with the SGLang inference engine. This tutorial demonstrated how to run SpecForge draft model training and test the model using SGLang on AMD GPUs. SpecForge itself is constantly evolving, and AMD is working closely with the open-source community to enhance this work for AMD platforms. This tutorial will hopefully encourage you to try, test, and contribute to SpecForge on AMD GPUs and help shape the future of AI acceleration.
