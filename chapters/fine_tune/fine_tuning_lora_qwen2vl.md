> 原文作者：Sergio Paniego；由 AMD 修改以支持 AMD GPU。  
> [AMD 官方教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/fine_tune/fine_tuning_lora_qwen2vl.html) · [固定版本源码](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/fine_tune/fine_tuning_lora_qwen2vl.ipynb)  
> 上游版本：`0071bfd4bd3bd9525405c09a6ae69c983b09312a` · 同步日期：2026-10-12  
> 翻译状态：机器初译，待逐篇人工校对；代码与命令保持原文，未在 AMD GPU 上运行验证。

[中文 Notebook](../../notebooks/fine_tune/fine_tuning_lora_qwen2vl.ipynb) · [英文原文](../../original/fine_tune/fine_tuning_lora_qwen2vl.md) · [中英对照](../../bilingual/fine_tune/fine_tuning_lora_qwen2vl.md)

---

<a id="fine-tuning-with-the-hugging-face-ecosystem-trl"></a>

# 使用 PEFT 微调视觉语言模型
**作者**：[Sergio Paniego](https://github.com/sergiopaniego)；由 [AMD](https://www.amd.com) 修改以支持 AMD GPU。
**知识水平**：中级

本Notebook演示了如何使用 Hugging Face 生态系统微调 [视觉语言模型 (VLM)](https://huggingface.co/blog/vlms)，特别是使用 [参数高效微调 (PEFT)](https://huggingface.co/docs/peft/index) 和 [Transformer 强化学习 (TRL)](https://huggingface.co/docs/trl/index) 库。

**注意**：此Notebook源自 [fine_tuning_vlm_trl](https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl)。

<a id="model-and-dataset-overview"></a>

## 模型和数据集概述

您将成为 [ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) 数据集上的 [Qwen2-VL-7B](https://qwenlm.github.io/blog/qwen2-vl/) 模型 微调。该数据集包含各种图表类型的图像与问答对，非常适合增强模型的视觉问答能力。

<a id="prerequisites"></a>

## 前置条件

本教程是使用以下设置进行开发和测试的。 

<a id="operating-system"></a>

### 操作系统

- **Ubuntu 22.04**：确保您的系统运行的是 Ubuntu 版本 22.04。

<a id="hardware"></a>

### 硬件

- **AMD Instinct™ GPU**：本教程在 AMD Instinct MI300X GPU 上进行了测试。确保您使用的是 AMD Instinct GPU 或支持 ROCm 的兼容硬件，并且您的系统满足[官方要求](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)。

<a id="software"></a>

### 软件

* **ROCm 6.2**：按照 [ROCm 安装指南](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html) 安装并验证 ROCm。安装后，使用以下命令确认您的设置：

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

<a id="weights--biases-api-access"></a>

### 权重和偏差 API 访问

* 从[权重与偏差(W&B)](https://wandb.ai/)获取API令牌。

<a id="data-preparation"></a>

### 数据准备

* 本教程使用 Hugging Face 中的示例数据集，该数据集是在设置步骤中准备的。

<a id="prepare-the-training-environment"></a>

## 准备训练环境

请按照以下步骤设置培训环境。

<a id="1-pull-the-docker-image"></a>

### 1. 拉取 Docker 镜像

确保您的系统满足[系统要求](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)。

拉取本教程所需的 Docker 映像：

``` bash
docker pull rocm/pytorch:rocm6.2.3_ubuntu22.04_py3.10_pytorch_release_2.3.0
```

<a id="2-launch-the-docker-container"></a>

### 2.启动Docker容器

运行此命令来启动 Docker 容器。

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
  --hostname=ROCm-FT \
  -v $(pwd):/workspace \
  -w /workspace/notebooks \
  rocm/pytorch:rocm6.2.3_ubuntu22.04_py3.10_pytorch_release_2.3.0
```

**注意**：该命令将当前目录挂载到容器中的`/workspace`目录中。确保在运行 Docker 命令之前将Notebook文件复制到此目录，或者在启动后上传到 Jupyter Notebook 环境。保存终端输出中提供的令牌或 URL，以便从 Web 浏览器访问Notebook。您可以从 [AI Developer Hub GitHub 存储库](https://github.com/ROCm/gpuaidev-docs) 下载此Notebook。


<a id="3-install-and-launch-jupyter"></a>

### 3.安装并启动 Jupyter

在 Docker 容器内，使用以下命令安装 Jupyter：

``` bash
pip install jupyter
```

启动 Jupyter 服务器：

```bash
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**注意**：在运行上述命令之前，请确保您的系统上尚未使用端口 `8888`。如果是，请通过将 `--port=8888` 替换为其他端口号来指定其他端口，例如 `--port=8890`。

<a id="4-install-the-dependencies"></a>

### 4.安装依赖项

验证 Torch 库是否已安装并且 GPU 是否可以访问。

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
print("Is a ROCm-GPU detected? ", torch.cuda.is_available())
print("How many ROCm-GPUs are detected? ", torch.cuda.device_count())
```

然后使用 `pip` 安装该库的以下依赖项。

```python
# Install necessary libraries for fine-tuning, including parameter-efficient fine-tuning (peft) and transformers
!pip install transformers==4.47.0 trl==0.12.0  peft==0.13.2 qwen-vl-utils==0.0.8 wandb==0.19.1 accelerate==1.1.1 ipywidgets==8.1.5 numpy>=1.26.4 numba
```

验证安装：

```python
# Check for the required libraries and their versions
!pip list | grep -E "transformers|trl|peft|qwen-vl-utils|wandb|accelerate|ipywidgets|numpy|numba"
```

<a id="5-provide-your-hugging-face-token"></a>

### 5. 提供您的拥抱令牌

登录 Hugging Face 上传您的微调模型。您需要使用 Hugging Face 帐户进行身份验证，才能直接从此Notebook保存和共享您的模型。

```python
from huggingface_hub import notebook_login

notebook_login()
```

验证您的令牌是否被正确接受：

```python
from huggingface_hub import HfApi

try:
    api = HfApi()
    user_info = api.whoami()
    print(f"Token validated successfully! Logged in as: {user_info['name']}")
except Exception as e:
    print(f"Token validation failed. Error: {e}")
```

<a id="load-the-dataset"></a>

## 加载数据集

在本部分中，您将加载 [HuggingFaceM4/ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) 数据集。该数据集包含与相关问题和答案配对的图表图像，非常适合视觉问答任务的训练。

接下来，为 VLM 生成系统消息。这创建了一个系统，该系统充当分析图表图像并根据这些图表提供问题的简洁答案的专家。

**⚠️重要：确保选择正确的内核**

如果验证过程失败，请确保为您的Notebook选择正确的 Jupyter 内核。
要更改内核，请执行以下步骤：

1. 进入 **内核** 菜单。
2. 选择**更改内核**
3. 从列表中选择 `Python 3 (ipykernel)`。

**未能选择正确的内核可能会导致运行Notebook时出现意外问题。**

```python
system_message = """You are a Vision Language Model specializing in interpreting visual data from chart images.
Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""
```

将数据集格式化为聊天机器人结构以进行交互。每次交互都包含一条系统消息，然后是图像和用户的查询，最后是查询的答案。

有关该型号的更多使用技巧，请参见[型号卡](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct#more-usage-tips)。

```python
def format_data(sample):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["image"],
                },
                {
                    "type": "text",
                    "text": sample['query'],
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": sample["label"][0]
                }
            ],
        },
    ]
```

出于教育目的，您只需加载数据集中每个分割的 10%。但是，在实际用例中，您通常会加载整组示例。

```python
from datasets import load_dataset

dataset_id = "HuggingFaceM4/ChartQA"
train_dataset, eval_dataset, test_dataset = load_dataset(dataset_id, split=['train[:10%]', 'val[:10%]', 'test[:10%]'])
```

接下来，查看数据集的结构。它包括图像、查询、标签（即答案）和您将丢弃的第四个特征。

```python
train_dataset
```

使用聊天机器人结构格式化数据。这会为模型适当地设置交互。

```python
train_dataset = [format_data(sample) for sample in train_dataset]
eval_dataset = [format_data(sample) for sample in eval_dataset]
test_dataset = [format_data(sample) for sample in test_dataset]
```

```python
train_dataset[200]
```

<a id="load-the-model-and-check-its-performance"></a>

## 加载模型并检查其性能

加载数据集后，加载模型并使用数据集中的示例评估其性能。本教程使用 [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)，这是一种能够理解视觉数据和文本的视觉语言模型 (VLM)。

```python
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

model_id = "Qwen/Qwen2-VL-7B-Instruct"
```

接下来，加载模型和分词器，为 推理 做好准备。

```python
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

processor = Qwen2VLProcessor.from_pretrained(model_id)
```

要评估模型的性能，请使用数据集中的样本。首先，检查该样本的内部结构。

```python
train_dataset[0]
```

使用不带系统消息的示例来评估 VLM 的原始理解。这是要使用的输入：

```python
train_dataset[0][1:2]
```

现在查看与示例相对应的图表。你能根据视觉信息回答问题吗？

```python
train_dataset[0][1]['content'][0]['image']
```

创建一个方法，将模型、处理器和样本作为输入来生成模型的答案。这使您能够简化 推理 流程并轻松评估 VLM 的性能。

```python
from qwen_vl_utils import process_vision_info

def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        sample[1:2],  # Use the sample without the system message
        tokenize=False,
        add_generation_prompt=True
    )

    # Process the visual input from the sample
    image_inputs, _ = process_vision_info(sample)

    # Prepare the inputs for the model
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(device)  # Move inputs to the specified device

    # Generate text with the model
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the output text
    output_text = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output_text[0]  # Return the first decoded output text
```

```python
# Example of how to call the method with sample:
output = generate_text_from_sample(model, processor, train_dataset[0], device="cuda")
output
```

虽然模型成功地检索了正确的视觉信息，但它很难准确地回答问题。这表明微调可能是增强其性能的关键。现在是时候继续 微调 流程了。

<a id="remove-the-model-and-clean-the-gpu"></a>

### 删除模型并清理GPU

在继续下一节中的模型训练之前，请清除当前变量并清理 GPU 以释放资源。

```python
import gc
import time

def clear_memory():
    # Delete variables if they exist in the current global scope
    if 'inputs' in globals(): del globals()['inputs']
    if 'model' in globals(): del globals()['model']
    if 'processor' in globals(): del globals()['processor']
    if 'trainer' in globals(): del globals()['trainer']
    if 'peft_model' in globals(): del globals()['peft_model']
    if 'bnb_config' in globals(): del globals()['bnb_config']
    time.sleep(2)

    # Garbage collection and clearing CUDA memory
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

clear_memory()
```

<a id="fine-tune-the-model-using-trl"></a>

## 使用 TRL 微调模型

请按照以下步骤微调您的模型。

<a id="1-load-the-model-for-training"></a>

### 1.加载模型进行训练

首先，加载原始模型。  

**注意**：或者，可以使用 [bitsandbytes](https://huggingface.co/docs/bitsandbytes/main/en/index) 加载量化模型。要了解有关 量化 的更多信息，请参阅 [Hugging Face 的这篇博文](https://huggingface.co/blog/merve/quantization) 或 [Maarten Grootendorst 的博文](https://www.maartengrootendorst.com/blog/quantization/)。

```python
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=torch.bfloat16
).half()
processor = Qwen2VLProcessor.from_pretrained(model_id)
```

<a id="2-set-up-lora-and-sftconfig"></a>

### 2. 设置 LoRA 和 SFTConfig

接下来，为训练设置配置 LoRA。 LoRA 通过应用低秩近似来减少内存使用，从而降低内存需求并提高训练效率，使其成为在不牺牲质量的情况下优化模型性能的绝佳选择。

```python
from peft import LoraConfig, get_peft_model

# Configure LoRA
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

# Apply PEFT model adaptation
peft_model = get_peft_model(model, peft_config)

# Print trainable parameters
peft_model.print_trainable_parameters()
```

使用监督 微调 (SFT) 来优化模型在任务上的性能。为此，请使用 [TRL 库](https://huggingface.co/docs/trl/index) 中的 [SFTConfig](https://huggingface.co/docs/trl/sft_trainer) 类定义训练参数。 SFT 提供标记数据，帮助模型学习根据其输入生成更准确的响应。这种方法可确保模型适合您的特定用例，从而在理解和响应视觉查询方面获得更好的性能。

```python
from trl import SFTConfig

# Configure training arguments
training_args = SFTConfig(
    output_dir="qwen2-7b-instruct-trl-sft-ChartQA",  # Directory to save the model
    num_train_epochs=3,  # Number of training epochs
    per_device_train_batch_size=4,  # Batch size for training
    per_device_eval_batch_size=4,  # Batch size for evaluation
    gradient_accumulation_steps=8,  # Steps to accumulate gradients
    gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
    # Optimizer and scheduler settings
    optim="adamw_torch_fused",  # Optimizer type
    # optim = "adamw_hf",
    learning_rate=2e-4,  # Learning rate for training
    lr_scheduler_type="constant",  # Type of learning rate scheduler
    # Logging and evaluation
    logging_steps=1,  # Steps interval for logging
    eval_steps=10,  # Steps interval for evaluation
    eval_strategy="steps",  # Strategy for evaluation
    save_strategy="steps",  # Strategy for saving the model
    save_steps=20,  # Steps interval for saving
    metric_for_best_model="eval_loss",  # Metric to evaluate the best model
    greater_is_better=False,  # Whether higher metric values are better
    load_best_model_at_end=True,  # Load the best model after training
    # Mixed precision and gradient settings
    bf16=False,  # Use bfloat16 precision
    fp16=True,  # Use float16 precision
    tf32=False,  # Use TensorFloat-32 precision
    max_grad_norm=0.3,  # Maximum norm for gradient clipping
    warmup_ratio=0.03,  # Ratio of total steps for warmup
    # Hub and reporting
    push_to_hub=False,  # Whether to push model to Hugging Face Hub, disable it by default.
    report_to=None,  # Reporting tool for tracking metrics
    # Gradient checkpointing settings
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
    # Dataset configuration
    dataset_text_field="",  # Text field in dataset
    dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
    #max_seq_length=1024  # Maximum sequence length for input
)

training_args.remove_unused_columns = False  # Keep unused columns in dataset
```

<a id="3-training-the-model"></a>

### 3. 训练模型

您可以使用 [权重和偏差 (W&B)](https://wandb.ai/) 记录您的训练进度。将Notebook连接到 W&B 以在培训期间捕获重要信息。

```python
import wandb

wandb.init(
    project="qwen2-7b-instruct-trl-sft-ChartQA",  # change this
    name="qwen2-7b-instruct-trl-sft-ChartQA",  # change this
    config=training_args,
)
```

该模型需要一个整理器函数来在训练过程中正确检索和批处理数据。此函数格式化模型的数据集输入，确保它们的结构正确。定义下面的整理器函数。

有关更多详细信息，请参阅 TRL 示例 [脚本](https://github.com/huggingface/trl/blob/main/examples/scripts/sft_vlm.py#L87)。

```python
# Create a data collator to encode text and image pairs
def collate_fn(examples):
    # Get the texts and images, and apply the chat template
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]  # Prepare texts for processing
    image_inputs = [process_vision_info(example)[0] for example in examples]  # Process the images to extract inputs

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)  # Encode texts and images into tensors

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()  # Clone input IDs for labels
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

    # Ignore the image token index in the loss computation (model specific)
    if isinstance(processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
        image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

    # Mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100  # Mask image token IDs in labels

    batch["labels"] = labels  # Add labels to the batch

    return batch  # Return the prepared batch
```

现在，定义 [SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)，它是 [Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) 类的包装器，并继承其属性和方法。当提供 [PeftConfig](https://huggingface.co/docs/peft/v0.6.0/en/package_reference/config#peft.PeftConfig) 对象时，此类通过正确初始化 [PeftModel](https://huggingface.co/docs/peft/v0.6.0/package_reference/peft_model) 来简化 微调 过程。通过使用`SFTTrainer`，您可以高效管理训练工作流程，并确保视觉语言模型获得流畅的微调体验。

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    peft_config=peft_config,
    tokenizer=processor.tokenizer,
)
```

现在是训练模型的时候了！

```python
trainer.train()
```

然后保存结果。

```python
trainer.save_model(training_args.output_dir)
```

<a id="testing-the-fine-tuned-model"></a>

## 测试微调模型

现在您已经成功地微调了视觉语言模型 (VLM)，是时候评估其性能了。本部分使用 ChartQA 数据集中的示例来测试模型，以了解它如何回答基于图表图像的问题。它提供了探索结果的好方法。

清理 GPU 内存以确保最佳性能：

```python
clear_memory()
```

然后使用与之前相同的管道重新加载基础模型。

```python
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

processor = Qwen2VLProcessor.from_pretrained(model_id)
```

将经过训练的适配器连接到预训练的模型。该适配器包含您在训练期间进行的 微调 调整，允许基础模型在不改变其核心参数的情况下利用新知识。集成适配器增强了模型的功能，同时保持了其原始结构。

```python
import os
os.getcwd()
```

```python
adapter_path = "./qwen2-7b-instruct-trl-sft-ChartQA"
model.load_adapter(adapter_path)
```

使用模型最初难以正确回答的数据集中的上一个样本。

```python
train_dataset[0][:2]
```

```python
train_dataset[0][1]['content'][0]['image']
```

```python
output = generate_text_from_sample(model, processor, train_dataset[0])
output
```

这个样本是从训练集中抽取的，因此模型在训练过程中遇到过它。这可以被视为一种作弊形式。为了更全面地了解模型的性能，请使用未见过的样本对其进行评估。

```python
test_dataset[10][:2]
```

```python
test_dataset[10][1]['content'][0]['image']
```

```python
output = generate_text_from_sample(model, processor, test_dataset[10])
output
```

该模型已成功学会响应数据集中指定的查询。你已经实现了你的目标！

<a id="compare-a-fine-tuned-model-versus-a-base-model-with-prompting"></a>

## 通过提示将微调模型与基本模型进行比较

您已经了解了 微调 VLM 如何成为使其适应您的特定需求的宝贵选择。另一种需要考虑的方法是直接使用提示或实施 RAG 系统，这在另一个[配方](https://huggingface.co/learn/cookbook/multimodal_rag_using_document_retrieval_and_vlms) 中有所介绍。

微调 VLM 需要大量数据和计算资源，这可能会产生成本。相反，您可以尝试提示，看看是否可以在没有 微调 开销的情况下实现类似的结果。

再次清理GPU内存以确保最佳性能。

```python
clear_memory()
```

首先，按照与之前相同的管道加载基线模型。

```python
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

processor = Qwen2VLProcessor.from_pretrained(model_id)
```

本案例再次使用之前的示例，但这次包含如下系统消息。此增强功能有助于将模型的输入置于上下文中，从而有可能提高其响应准确性。

```python
train_dataset[0][:2]
```

现在看看它的表现如何：

```python
text = processor.apply_chat_template(
    train_dataset[0][:2], tokenize=False, add_generation_prompt=True
)

image_inputs, _ = process_vision_info(train_dataset[0])

inputs = processor(
    text=[text],
    images=image_inputs,
    return_tensors="pt",
)

inputs = inputs.to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]

output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

output_text[0]
```

如图所示，该模型使用预训练模型以及附加系统消息生成正确答案，无需任何训练。根据具体用例，此方法可以作为 微调 的可行替代方案。
