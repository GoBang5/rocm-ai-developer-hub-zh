> 原文作者：ModelScope and Tongyi Lab (Alibaba Group)  
> [AMD 官方教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/fine_tune/qwen_image.html) · [固定版本源码](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/fine_tune/qwen_image.ipynb)  
> 上游版本：`0071bfd4bd3bd9525405c09a6ae69c983b09312a` · 同步日期：2026-10-12  
> 翻译状态：机器初译，待逐篇人工校对；代码与命令保持原文，未在 AMD GPU 上运行验证。

[中文 Notebook](../../notebooks/fine_tune/qwen_image.ipynb) · [英文原文](../../original/fine_tune/qwen_image.md) · [中英对照](../../bilingual/fine_tune/qwen_image.md)

---

<a id="customize-qwen-image-with-diffsynth-studio"></a>

# 使用 DiffSynth-Studio 定制 Qwen-Image
**作者**：ModelScope and [Tongyi Lab](https://www.linkedin.com/company/alibaba-tongyi-lab/) (Alibaba Group)
**知识水平**：中级

本教程探讨了 [Qwen-Image](https://qwen-image.org/) 系列（一个庞大的 86B 参数模型集合）的功能，并解释了如何在 AMD 硬件上使用 [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) 对其进行有效微调。它演示了 AMD Instinct™ MI300X GPU 的高内存容量如何能够同时加载多个大型模型，以执行涉及 推理、编辑和训练的复杂工作流程。

**注**：本教程由ModelScope和[统一实验室](https://www.linkedin.com/company/alibaba-tongyi-lab/)(Alibaba组）共同开发。 

<a id="key-components"></a>

## 关键组件
* **硬件**：AMD Instinct MI300X GPU
* **软件**：[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) 和 [ROCm](https://rocm.docs.amd.com/en/latest/index.html)
* **型号**：[Qwen-Image](https://qwen-image.org/)、[Qwen-Image-Edit](https://www.modelscope.cn/models/Qwen/Qwen-Image-Edit) 和自定义 LoRA 适配器

<a id="prerequisites"></a>

## 前置条件
开始之前，请确保您的环境满足以下要求：
* **操作系统**：Linux（推荐Ubuntu 22.04）。有关支持的操作系统，请参阅[官方要求](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)。
* **硬件**：AMD Instinct MI300X GPU
* **软件**：ROCm 6.0 或更高版本、Docker 和 Python 3.10 或更高版本

**注意**：按照[ROCm安装指南](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html)安装并验证ROCm。

<a id="step-1-environment-setup"></a>

## 步骤1：环境设置

请按照以下步骤设置本教程的环境。

<a id="verify-the-hardware-availability"></a>

### 验证硬件可用性
AMD Instinct MI300X GPU 旨在为生成式 AI 工作负载提供峰值性能。在开始之前，请验证您的 GPU 是否已正确检测到并可供使用。

```python
!amd-smi
#For ROCm 6.4 and earlier, run rocm-smi instead.
```

<a id="install-diffsynth-studio-from-source"></a>

### 从源代码安装 DiffSynth-Studio
为了确保与 AMD ROCm 完全兼容，请直接从源安装 [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)。 

**注意：** 安装后，手动更新系统路径，以确保Notebook可以立即导入库，而无需重新启动内核。

```python
import os
import sys

# 1. Clone the repository
!git clone https://github.com/modelscope/DiffSynth-Studio.git

# 2. Navigate into the directory
os.chdir("DiffSynth-Studio")

# 3. Checkout the specific commit for reproducibility
!git checkout afd101f3452c9ecae0c87b79adfa2e22d65ffdc3

# 4. Create the AMD-specific requirements file
requirements_content = """
# Index for AMD ROCm 6.4 wheels (Prioritized)
--index-url https://download.pytorch.org/whl/rocm6.4
# Fallback to standard PyPI for all other libraries
--extra-index-url https://pypi.org/simple
# Core PyTorch libraries
torch>=2.0.0
torchvision
# Install the DiffSynth-Studio project and its other dependencies
-e .
""".strip()

with open("requirements-amd.txt", "w") as f:
    f.write(requirements_content)

# 5. Install using the custom requirements
!pip install -r requirements-amd.txt

# 6. Force the current notebook to see the installed package
sys.path.append(os.getcwd())
print(f"Added {os.getcwd()} to system path to enable immediate import.")

# 7. Return to root directory
os.chdir("..")
```

<a id="step-2-basic-model-inference"></a>

## 步骤2：基本型号推理

本节演示如何使用模型执行 推理。

<a id="load-qwen-image"></a>

### 加载 Qwen-Image
[Qwen-Image](https://www.modelscope.ai/models/Qwen/Qwen-Image)是一个大规模图像生成模型。配置管道并将模型组件（Transformer、Text Encoder 和 VAE）加载到 GPU 上。

**注意**：配置环境以使用 ModelScope 作为下载权重的域。

```python
import warnings
warnings.filterwarnings("ignore")
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MODELSCOPE_DOMAIN"] = "www.modelscope.ai"
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from modelscope import dataset_snapshot_download
import torch
from PIL import Image
import pandas as pd
import numpy as np
qwen_image = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)
qwen_image.enable_lora_magic()
```

<a id="generate-a-baseline-image"></a>

### 生成基线图像

使用简单的提示生成您的第一张图像：*“美丽的亚洲女人的肖像”*。

```python
prompt = "a portrait of a beautiful Asian woman"
image = qwen_image(prompt, seed=0, num_inference_steps=40)
image.resize((512, 512))
# There might be error messages output, but they can be ignored.
```

<a id="step-3-enhancing-quality-with-lora"></a>

## 步骤 3：使用 LoRA 提高质量
您可能会注意到基线图像缺乏精细的细节。 

要改进图像，请加载 [Qwen-Image-LoRA-ArtAug-v1](https://www.modelscope.ai/models/DiffSynth-Studio/Qwen-Image-LoRA-ArtAug-v1) 以显着增强生成图像中的视觉保真度和艺术细节。

```python
qwen_image.load_lora(
    qwen_image.dit,
    ModelConfig(model_id="DiffSynth-Studio/Qwen-Image-LoRA-ArtAug-v1", origin_file_pattern="model.safetensors"),
    hotload=True,
)
```

重新运行相同的提示以查看改进。

```python
prompt = "a portrait of a beautiful Asian woman"
image = qwen_image(prompt, seed=0, num_inference_steps=40)
image.save("image_face.jpg")
image.resize((512, 512))
```

<a id="step-4-advanced-image-editing"></a>

## 步骤 4：高级图像编辑

本节介绍一些用于生成更复杂图像的高级技术。

<a id="load-the-editing-pipeline"></a>

### 加载编辑管道
Qwen-Image 系列包括适用于不同任务的专用模型。接下来，加载[Qwen-Image-Edit](https://www.modelscope.cn/models/Qwen/Qwen-Image-Edit)，这是一个专为图像编辑和修复任务设计的模型。

```python
qwen_image_edit = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
)
qwen_image_edit.enable_lora_magic()
```

<a id="outpainting-with-consistency"></a>

### 保持一致的涂色

您可以通过将刚刚生成的肖像并将其扩展为具有森林背景的长镜头图像来执行绘制任务。

```python
prompt = "Realistic photography of a beautiful woman wearing a long dress. The background is a forest."
negative_prompt = "Make the character's fingers mutilated and distorted, enlarge the head to create an unnatural head-to-body ratio, turning the figure into a short-statured big-headed doll. Generate harsh, glaring sunlight and render the entire scene with oversaturated colors. Twist the legs into either X-shaped or O-shaped deformities."
image = qwen_image_edit(prompt, negative_prompt=negative_prompt, edit_image=Image.open("image_face.jpg"), seed=1, num_inference_steps=40)
image.resize((512, 512))
```

这张照片中的面孔显得不一致。加载专用 LoRA 模型 [DiffSynth-Studio/Qwen-Image-Edit-F2P](https://www.modelscope.ai/models/DiffSynth-Studio/Qwen-Image-Edit-F2P)，该模型可以根据面部参考生成一致的图像。

```python
qwen_image_edit.load_lora(
    qwen_image_edit.dit,
    ModelConfig(model_id="DiffSynth-Studio/Qwen-Image-Edit-F2P", origin_file_pattern="model.safetensors"),
    hotload=True,
)
prompt = "Realistic photography of a beautiful woman wearing a long dress. The background is a forest."
negative_prompt = "Make the character's fingers mutilated and distorted, enlarge the head to create an unnatural head-to-body ratio, turning the figure into a short-statured big-headed doll. Generate harsh, glaring sunlight and render the entire scene with oversaturated colors. Twist the legs into either X-shaped or O-shaped deformities."
image = qwen_image_edit(prompt, negative_prompt=negative_prompt, edit_image=Image.open("image_face.jpg"), seed=1, num_inference_steps=40)
image.save("image_fullbody.jpg")
image.resize((512, 512))
```

<a id="step-5-multilingual-and-multi-image-editing"></a>

## 步骤5：多语言和多图像编辑

Qwen-Image 文本编码器足够强大，可以理解未经过明确训练的语言的提示。要尝试此操作，请使用韩语提示生成一个字符。首先，使用英文生成图像。

```python
qwen_image.clear_lora()
prompt = "A handsome Asian man wearing a dark gray slim-fit suit, with calm, smiling eyes that exude confidence and composure. He is seated at a table, holding a bouquet of red flowers in his hands."
image = qwen_image(prompt, seed=2, num_inference_steps=40)
image.resize((512, 512))
```

然后用韩语判断模型是否能够理解图像内容。

```python
qwen_image.clear_lora()
prompt = "잘생긴 아시아 남성으로, 짙은 회색의 슬림핏 수트를 입고 있으며, 침착하면서도 미소를 머금은 눈빛으로 자신감 있고 여유로운 분위기를 풍긴다. 그는 책상 앞에 앉아 붉은 꽃다발을 손에 들고 있다."
image = qwen_image(prompt, seed=2, num_inference_steps=40)
image.save("image_man.jpg")
image.resize((512, 512))
```

尽管 Qwen-Image 没有接受韩语文本训练，但其文本编码器的基本功能仍然提供多语言理解。

<a id="merging-subjects-with-qwen-image-edit-2509"></a>

### 使用 Qwen-Image-Edit-2509 合并主题

现在您有两个图像：森林里的女人和拿着鲜花的男人。使用支持多图像编辑的 [Qwen-Image-Edit-2509](https://www.modelscope.cn/models/Qwen/Qwen-Image-Edit-2509)，您可以将这两个独立的图像合并为一个角色互动的连贯场景。

```python
qwen_image_edit_2509 = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
)
qwen_image_edit_2509.enable_lora_magic()
```

现在，生成一张这两个人在一起的照片。

```python
prompt = "이 사랑 넘치는 부부의 포옹하는 모습을 찍은 사진을 생성해 줘."
image = qwen_image_edit_2509(prompt, edit_image=[Image.open("image_fullbody.jpg"), Image.open("image_man.jpg")], seed=3, num_inference_steps=40)
image.save("image_merged.jpg")
image.resize((512, 512))
```

<a id="step-6-the-power-of-the-instinct-mi300x"></a>

## 步骤 6：Instinct MI300X 的强大功能
您当前已将三个大型模型同时加载到内存中。计算总参数计数以了解此工作负载的规模。

```python
def count_parameters(model):
    return sum([p.numel() for p in model.parameters()])

print(count_parameters(qwen_image) + count_parameters(qwen_image_edit) + count_parameters(qwen_image_edit_2509))
```

**总参数**：约 860 亿。

在标准 GPU 上处理这个问题是不可能的。然而，AMD Instinct MI300X GPU 具有 192 GB 的 VRAM，因此它可以将所有这些模型保留在内存中，以便在 推理、编辑和训练任务之间无缝切换。

```python
!amd-smi
#For ROCm 6.4 and earlier, run rocm-smi instead.
```

<a id="step-7-training-a-custom-lora"></a>

## 步骤 7：训练自定义 LoRA
最后，是时候从 推理 转向训练了。训练自定义 LoRA 适配器来向模型教授特定概念，在本例中为特定的狗。

<a id="prepare-the-dataset"></a>

### 准备数据集
下载包含五张狗图像和相关元数据的小数据集。

```python
!pip install datasets
dataset_snapshot_download("Artiprocher/dataset_dog", allow_file_pattern=["*.jpg", "*.csv"], local_dir="dataset")
images = [Image.open(f"dataset/{i}.jpg") for i in range(1, 6)]
Image.fromarray(np.concatenate([np.array(image.resize((256, 256))) for image in images], axis=1))
```

这是该数据集的元数据，包括带注释的图像描述。

```python
pd.read_csv("dataset/metadata.csv")
```

在训练之前，请验证提示符 `"a dog"` 的基本模型的输出。正如预期的那样，它会生成一只通用的狗，而不是您的特定主题。

```python
qwen_image.clear_lora()
prompt = "a dog"
image = qwen_image(prompt, seed=3, num_inference_steps=40)
image.resize((512, 512))
```

<a id="run-the-training-script"></a>

### 运行训练脚本
首先，清理一些 GPU 内存，为训练过程腾出空间。然后下载官方训练脚本并使用 `accelerate` 命令启动它。

```python
del qwen_image
del qwen_image_edit
del qwen_image_edit_2509
torch.cuda.empty_cache()
```

下载训练脚本。

```python
!wget https://github.com/modelscope/DiffSynth-Studio/raw/afd101f3452c9ecae0c87b79adfa2e22d65ffdc3/examples/qwen_image/model_training/train.py
```

运行训练任务。

```python
cmd = rf"""
accelerate launch train.py \
  --dataset_base_path dataset \
  --dataset_metadata_path dataset/metadata.csv \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "Qwen/Qwen-Image:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "lora_dog" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 32 \
  --dataset_num_workers 2 \
  --find_unused_parameters
""".strip()
os.system(cmd)
```

<a id="step-8-inference-with-the-custom-lora"></a>

## 步骤 8：使用定制 LoRA 的 推理
现在训练已完成，再次加载模型，注入新训练的 `lora_dog`，并验证模型是否识别您的特定狗。

```python
qwen_image = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)
qwen_image.enable_lora_magic()
```

接下来，重新加载模型并生成狗的照片。

```python
qwen_image.load_lora(
    qwen_image.dit,
    "lora_dog/epoch-0.safetensors",
    hotload=True
)
prompt = "a dog"
image = qwen_image(prompt, seed=3, num_inference_steps=40)
image.resize((512, 512))
```

生成另一张狗的图像。

```python
prompt = "a dog is jumping."
image = qwen_image(prompt, seed=3, num_inference_steps=40)
image.resize((512, 512))
```

<a id="conclusion"></a>

## 结论
本教程演示了 AMD Instinct MI300X 的端到端功能。您使用具有 86B 集体参数的模型成功执行了 推理，以高一致性编辑了图像，并训练了自定义适配器，所有这些都在单个 GPU 上完成。
