> 原文作者：Unsloth；由 AMD 修改以支持 AMD GPU。  
> [AMD 官方教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/fine_tune/unsloth_Llama3_1_8B_GRPO.html) · [固定版本源码](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/fine_tune/unsloth_Llama3_1_8B_GRPO.ipynb)  
> 上游版本：`0071bfd4bd3bd9525405c09a6ae69c983b09312a` · 同步日期：2026-10-12  
> 翻译状态：机器初译，待逐篇人工校对；代码与命令保持原文，未在 AMD GPU 上运行验证。

<!-- cell:1 -->

<a id="train-your-own-r1-reasoning-model-with-unsloth"></a>

# 使用 Unsloth 进行 GRPO 训练
**作者**：[Unsloth](https://unsloth.ai)；由 [AMD](https://www.amd.com) 修改以支持 AMD GPU。
**知识水平**：中级

本教程演示如何利用 [Unsloth](https://unsloth.ai) 在 AMD ROCm GPU 上微调 Llama-3.1 8B 大语言模型 (LLM)。 DeepSeek 的 R1 研究揭示了一个“顿悟时刻”，R1-Zero 通过使用 组相对策略优化 (GRPO)，在没有人类反馈的情况下自主学习分配更多思考时间。 Unsloth 团队增强了整个 GRPO 流程，使其使用的 VRAM 比 Hugging Face 和 Flash Attention 2 (FA2) 少 80%。这让您可以使用 Qwen2.5 (1.5B) 在仅 7GB VRAM 上重现 R1-Zero 的成就。

<details markdown="1">
<summary>查看英文原文 · 单元 1</summary>

# Train your own R1 reasoning model with Unsloth

**Authored by**: [Unsloth](https://unsloth.ai) and modified by [AMD](https://www.amd.com) to run on AMD GPUs.  
**Knowledge level**: Intermediate

This tutorial demonstrates how to fine-tune the Llama-3.1 8B large language model (LLM) on AMD ROCm GPUs by leveraging [Unsloth](https://unsloth.ai). DeepSeek’s R1 research revealed an “aha moment” where R1-Zero autonomously learned to allocate more thinking time without human feedback by using Group Relative Policy Optimization (GRPO). The Unsloth team has enhanced the entire GRPO process, making it use 80% less VRAM than Hugging Face and Flash Attention 2 (FA2). This lets you reproduce R1-Zero's achievement on just 7GB of VRAM using Qwen2.5 (1.5B).

</details>

<!-- cell:2 -->

<a id="prerequisites"></a>

## 前置条件

本教程是使用以下设置进行开发和测试的。 

<a id="operating-system"></a>

### 操作系统

* **Ubuntu 22.04**：确保您的系统运行的是 Ubuntu 版本 22.04。

<a id="hardware"></a>

### 硬件

* **AMD Instinct™ GPU**：本教程在 AMD Instinct MI300X GPU 上进行了测试。确保您使用的是 AMD Instinct GPU 或支持 ROCm 的兼容硬件，并且您的系统满足[官方要求](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)。

<a id="software"></a>

### 软件

* **ROCm 6.3**：按照 [ROCm 安装指南](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html) 安装并验证 ROCm。安装后，使用以下命令确认您的设置：

    ``` bash
    amd-smi
    ```

    此命令列出了您的 AMD GPU 以及相关详细信息。
    
    **注意**：对于 ROCm 6.4 及更早版本，请改用 `rocm-smi` 命令。


* **Docker**：确保 Docker 已正确安装和配置。请遵循适用于您的操作系统的 Docker 安装指南。

   **注意**：确保正确配置 Docker 权限。要配置权限以允许非 root 访问，请运行以下命令：

   ``` bash
   sudo usermod -aG docker $USER
   newgrp docker
   ```

   验证 Docker 是否正常工作：

   ``` bash
   docker run hello-world
   ```

<a id="hugging-face-api-access"></a>

### Hugging Face API 访问

* 从[Hugging Face](https://huggingface.co)获取API令牌用于下载模型。
* 确保 Hugging Face API 令牌具有访问 [Meta Llama 检查点](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) 所需的权限和批准。

<a id="data-preparation"></a>

### 数据准备

* 本教程使用 Hugging Face 中的示例数据集，该数据集是在设置步骤中准备的。

<details markdown="1">
<summary>查看英文原文 · 单元 2</summary>

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

</details>

<!-- cell:3 -->

<a id="prepare-the-training-environment"></a>

## 准备训练环境

<a id="1-pull-the-docker-image"></a>

### 1. 拉取 Docker 镜像

确保您的系统满足[系统要求](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)。

拉取本教程所需的 Docker 映像：

``` bash
docker pull rocm/vllm-dev:main
```

<a id="2-launch-the-docker-container"></a>

### 2.启动Docker容器

启动 Docker 容器并映射必要的目录。将 `/path/to/notebooks` 替换为主机上存储这些Notebook的目录的完整路径。

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
  rocm/vllm-dev:main
```

**注意**：该命令将当前目录挂载到容器中的`/workspace`目录中。确保在运行 Docker 命令之前将Notebook文件复制到此目录，或者在启动后上传到 Jupyter Notebook 环境。保存终端输出中提供的令牌或 URL，以便从 Web 浏览器访问Notebook。您可以从 [AI Developer Hub GitHub 存储库](https://github.com/ROCm/gpuaidev) 下载此Notebook。

<a id="3-install-and-launch-jupyter"></a>

### 3.安装并启动 Jupyter

在 Docker 容器内，使用以下命令安装 Jupyter：

``` bash
pip install jupyter
```

启动 Jupyter 服务器：

``` bash
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**注意**：在运行上述命令之前，请确保您的系统上尚未使用端口 `8888`。如果是，您可以通过将 `--port=8888` 替换为其他端口号来指定其他端口，例如 `--port=8890`。

<details markdown="1">
<summary>查看英文原文 · 单元 3</summary>

## Prepare the training environment

### 1. Pull the Docker image

Ensure your system meets the [system requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

Pull the Docker image required for this tutorial:

``` bash
docker pull rocm/vllm-dev:main
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
  rocm/vllm-dev:main
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

</details>

<!-- cell:4 -->

<a id="4-install-the-required-libraries"></a>

### 4.安装所需的库

安装本教程所需的库。在 Docker 容器中运行的 Jupyter Notebook 中运行以下命令：

<details markdown="1">
<summary>查看英文原文 · 单元 4</summary>

### 4. Install the required libraries

Install the libraries required for this tutorial. Run the following commands inside the Jupyter notebook running within the Docker container:

</details>

```python
# Install Unsloth from source
!git clone https://github.com/billishyahao/unsloth.git && cd unsloth && git checkout billhe/rocm &&  pip install .
!pip install unsloth_zoo==2025.3.17
# Install ROCm Bitsandbytes from source 
!git clone --recurse https://github.com/ROCm/bitsandbytes && cd bitsandbytes && git checkout rocm_enabled_multi_backend && pip install -r requirements-dev.txt && cmake -DCOMPUTE_BACKEND=hip -S . && make -j  && pip install .
# This notebook is verified under unsloth==2025.3.19 unsloth_zoo==2025.3.17 bitsandbytes==0.43.3.dev0
```

<!-- cell:6 -->

验证安装：

<details markdown="1">
<summary>查看英文原文 · 单元 6</summary>

Verify the installation:

</details>

```python
# Verify the installation and version of the required libraries
!pip list | grep unsloth
```

<!-- cell:8 -->

这是预期的输出：

```
unsloth                           2025.3.19
unsloth_zoo                       2025.3.17
```

<details markdown="1">
<summary>查看英文原文 · 单元 8</summary>

Here is the expected output:

```
unsloth                           2025.3.19
unsloth_zoo                       2025.3.17
```

</details>

<!-- cell:9 -->

**⚠️重要**：确保选择正确的内核

如果验证过程失败，请确保为您的Notebook选择正确的 Jupyter 内核。
要更改内核，请执行以下步骤：

1. 进入 **内核** 菜单。
2. 选择**更改内核**。
3. 从列表中选择 `Python 3 (ipykernel)`。

**重要**：未能选择正确的内核可能会导致运行Notebook时出现意外问题。

<details markdown="1">
<summary>查看英文原文 · 单元 9</summary>

**⚠️ Important**: Ensure the correct kernel is selected

If the verification process fails, ensure the correct Jupyter kernel is selected for your notebook.
To change the kernel, follow these steps:

1. Go to the **Kernel** menu.
2. Select **Change Kernel**.
3. Select `Python 3 (ipykernel)` from the list.

**Important**: Failure to select the correct kernel can lead to unexpected issues when running the notebook.

</details>

<!-- cell:10 -->

<a id="5-provide-your-hugging-face-token"></a>

### 5. 提供您的拥抱令牌

您需要 Hugging Face API 令牌才能访问 Llama-3.1。在 [Hugging Face Tokens](https://huggingface.co/settings/tokens) 生成您的令牌并请求访问 [Llama-3.1 8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)。令牌通常以“hf_”开头。 

在 Jupyter Notebook中运行以下交互式块来设置令牌：

**注意**：取消选中“添加令牌作为 Git 凭证”选项。

<details markdown="1">
<summary>查看英文原文 · 单元 10</summary>

### 5. Provide your Hugging Face token

You'll require a Hugging Face API token to access Llama-3.1. Generate your token at [Hugging Face Tokens](https://huggingface.co/settings/tokens) and request access for [Llama-3.1 8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct). Tokens typically start with "hf_". 

Run the following interactive block in your Jupyter notebook to set up the token:

**Note**: Uncheck the "Add token as Git credential" option.

</details>

```python
from huggingface_hub import notebook_login, HfApi

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

<!-- cell:13 -->

<a id="running-grpo"></a>

## 运行 GRPO

按照以下步骤准备数据、训练模型并执行 推理。

<details markdown="1">
<summary>查看英文原文 · 单元 13</summary>

## Running GRPO

Follow these steps to prepare the data, train the model, and perform inference.

</details>

<!-- cell:14 -->

<a id="set-the-parameters"></a>

### 设置参数

加载`Llama 3.1 8B Instruct`并设置参数：

<details markdown="1">
<summary>查看英文原文 · 单元 14</summary>

### Set the parameters

Load `Llama 3.1 8B Instruct` and set the parameters:

</details>

```python
from unsloth import FastLanguageModel
import torch
max_seq_length = 1024 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/meta-Llama-3.1-8B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # True for LoRA 4bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)
```

<!-- cell:16 -->

<a id="data-preparation"></a>

### 数据准备
<a name="Data"></a>

本教程直接利用 [willccbb](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb) 进行数据准备和所有奖励功能。您可以自由创建自己的方法。

<details markdown="1">
<summary>查看英文原文 · 单元 16</summary>

### Data preparation
<a name="Data"></a>

This tutorial directly leverages [willccbb](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb) for data preparation and all reward functions. You are free to create your own method.

</details>

```python
import re
from datasets import load_dataset, Dataset

# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

dataset = get_gsm8k_questions()

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]
```

<!-- cell:18 -->

<a name="Train"></a>
<a id="train-the-model"></a>

### 训练模型

现在设置 GRPO Trainer 和所有配置：

<details markdown="1">
<summary>查看英文原文 · 单元 18</summary>

<a name="Train"></a>
### Train the model

Now set up the GRPO Trainer and all configurations:

</details>

```python
max_prompt_length = 256

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 6, # Decrease if out of memory
    max_prompt_length = max_prompt_length,
    max_completion_length = max_seq_length - max_prompt_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 250,
    save_steps = 250,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",
)
```

<!-- cell:20 -->

现在您可以运行训练器了。向上滚动查看奖励表。目标是看到`reward`栏增加！

您可能需要等待 150 到 200 步才能看到任何操作。前 100 步您可能不会获得任何奖励。请耐心等待！

|步骤|训练损失|奖励 |奖励标准 |完成长度 |吉隆坡 |
|------|---------------|-----------|------------|-------------------|----------|
| 1 | 0.000000 | 0.000000 0.125000 | 0.000000 | 0.000000 200.000000 | 0.000000 | 0.000000
| 2 | 0.000000 | 0.000000 0.072375 | 0.072375 0.248112 | 200.000000 | 0.000000 | 0.000000
| 3 | 0.000000 | 0.000000 -0.079000 | -0.079000 0.163776 | 182.500000 | 0.000005 | 0.000005

<details markdown="1">
<summary>查看英文原文 · 单元 20</summary>

Now you can run the trainer. Scroll up to see a table of rewards. The goal is to see the `reward` column increase!

You might have to wait for 150 to 200 steps to see any action. You probably won't get any reward for the first 100 steps. Please be patient!

| Step | Training loss | reward    | reward_std | completion_length | kl       |
|------|---------------|-----------|------------|-------------------|----------|
| 1    | 0.000000      | 0.125000  | 0.000000   | 200.000000        | 0.000000 |
| 2    | 0.000000      | 0.072375  | 0.248112   | 200.000000        | 0.000000 |
| 3    | 0.000000      | -0.079000 | 0.163776   | 182.500000        | 0.000005 |

</details>

```python
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()
```

<!-- cell:22 -->

<a name="Inference"></a>
<a id="inference"></a>

### 推理
现在尝试您刚刚训练的模型。首先尝试没有任何 GRPO 训练的模型：

<details markdown="1">
<summary>查看英文原文 · 单元 22</summary>

<a name="Inference"></a>
### Inference
Now try the model you just trained. First try the model without any GRPO training:

</details>

```python
text = tokenizer.apply_chat_template([
    {"role" : "user", "content" : "Calculate pi."},
], tokenize = False, add_generation_prompt = True)

from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature = 0.8,
    top_p = 0.95,
    max_tokens = 1024,
)
output = model.fast_generate(
    [text],
    sampling_params = sampling_params,
    lora_request = None,
)[0].outputs[0].text

output
```

<!-- cell:24 -->

现在尝试使用刚刚使用 GRPO 训练的 LoRA，但首先保存 LoRA。

<details markdown="1">
<summary>查看英文原文 · 单元 24</summary>

Now try it with the LoRA you just trained with GRPO, but save the LoRA first.

</details>

```python
model.save_lora("grpo_saved_lora")
```

<!-- cell:26 -->

现在您可以加载 LoRA 并进行测试：

<details markdown="1">
<summary>查看英文原文 · 单元 26</summary>

Now you can load the LoRA and test:

</details>

```python
text = tokenizer.apply_chat_template([
    {"role" : "system", "content" : SYSTEM_PROMPT},
    {"role" : "user", "content" : "Calculate pi."},
], tokenize = False, add_generation_prompt = True)

from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature = 0.8,
    top_p = 0.95,
    max_tokens = 1024,
)
output = model.fast_generate(
    text,
    sampling_params = sampling_params,
    lora_request = model.load_lora("grpo_saved_lora"),
)[0].outputs[0].text

output
```

<!-- cell:28 -->

推理模型要好得多。它并不总是正确的，因为您只训练了一个小时左右。如果你延长序列长度并训练更长的时间，那就更好了。

<details markdown="1">
<summary>查看英文原文 · 单元 28</summary>

The reasoning model is much better. It's not always correct, since you only trained it for an hour or so. It'll be better if you extend the sequence length and train it for longer.

</details>

<!-- cell:29 -->

<a name="Save"></a>
<a id="saving-to-float16-for-vllm"></a>

### 保存为 VLLM 的 float16

Unsloth还支持直接保存到`float16`。选择 `merged_16bit` 为 `float16` 或 `merged_4bit` 为 `int4`。它还允许 `lora` 适配器作为后备。使用 `push_to_hub_merged` 上传到您的 Hugging Face 帐户。访问[抱脸令牌设置](https://huggingface.co/settings/tokens)获取您的个人令牌。

<details markdown="1">
<summary>查看英文原文 · 单元 29</summary>

<a name="Save"></a>
### Saving to float16 for VLLM

Unsloth also supports saving to `float16` directly. Select `merged_16bit` for `float16` or `merged_4bit` for `int4`. It also allows `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account. Visit the [Hugging Face token settings](https://huggingface.co/settings/tokens) for your personal tokens.

</details>

```python
# Merge to 16bit
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

# Merge to 4bit
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

# Just LoRA adapters
if False: model.save_pretrained_merged("model", tokenizer, save_method = "lora",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "lora", token = "")
```

<!-- cell:31 -->

现在你完成了！如果您对 Unsloth 有任何疑问、需要帮助或想要保持更新，他们有一个 [Discord](https://discord.gg/unsloth) 频道和一个 [GitHub](https://github.com/unslothai/unsloth)。您还可以查阅他们的[文档](https://docs.unsloth.ai/)以获取更多信息。

<details markdown="1">
<summary>查看英文原文 · 单元 31</summary>

Now you're done! If you have any questions about Unsloth, need help, or want to keep updated, they have a [Discord](https://discord.gg/unsloth) channel and a [GitHub](https://github.com/unslothai/unsloth). You can also consult their [documentation](https://docs.unsloth.ai/) for more information.

</details>
