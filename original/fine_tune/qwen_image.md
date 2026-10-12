> [Upstream source](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/fine_tune/qwen_image.ipynb) · Snapshot: `0071bfd4bd3bd9525405c09a6ae69c983b09312a`

# Customize Qwen-Image with DiffSynth-Studio

**Authored by**: ModelScope and [Tongyi Lab](https://www.linkedin.com/company/alibaba-tongyi-lab/) (Alibaba Group)  
**Knowledge level**: Intermediate

This tutorial explores the capabilities of the [Qwen-Image](https://qwen-image.org/) series—a massive 86B parameter model collection—and explains how to fine-tune it efficiently using [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) on AMD hardware. It demonstrates how the high memory capacity of the AMD Instinct™ MI300X GPU enables loading multiple large models simultaneously for complex workflows involving inference, editing, and training.

**Note**: This tutorial was developed by ModelScope and [Tongyi Lab](https://www.linkedin.com/company/alibaba-tongyi-lab/)(Alibaba Group). 

## Key components
* **Hardware**: AMD Instinct MI300X GPU
* **Software**: [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) and [ROCm](https://rocm.docs.amd.com/en/latest/index.html)
* **Models**: [Qwen-Image](https://qwen-image.org/), [Qwen-Image-Edit](https://www.modelscope.cn/models/Qwen/Qwen-Image-Edit), and Custom LoRA adapters

## Prerequisites
Before starting, ensure your environment meets the following requirements:
* **Operating system**: Linux (Ubuntu 22.04 recommended). See the [official requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) for supported operating systems.
* **Hardware**: AMD Instinct MI300X GPU
* **Software**: ROCm 6.0 or later, Docker, and Python 3.10 or later

**Note**: Install and verify ROCm by following the [ROCm install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html).

## Step 1: Environment setup

Follow these steps to set up your environment for the tutorial.

### Verify the hardware availability
The AMD Instinct MI300X GPU is designed to deliver peak performance for Generative AI workloads. Before you begin, verify that your GPU is correctly detected and ready for use.

```python
!amd-smi
#For ROCm 6.4 and earlier, run rocm-smi instead.
```

### Install DiffSynth-Studio from source
To ensure full compatibility with AMD ROCm, install [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) directly from the source. 

**Note:** After installation, manually update the system path to ensure the notebook can import the library immediately without a kernel restart.

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

## Step 2: Basic model inference

This section demonstrates how to conduct inference with the model.

### Load Qwen-Image
[Qwen-Image](https://www.modelscope.ai/models/Qwen/Qwen-Image) is a large-scale image generation model. Configure the pipeline and load the model components (Transformer, Text Encoder, and VAE) onto the GPU.

**Note**: Configure the environment to use ModelScope as the domain for downloading weights.

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

### Generate a baseline image

Generate your first image using the simple prompt: *"a portrait of a beautiful Asian woman"*.

```python
prompt = "a portrait of a beautiful Asian woman"
image = qwen_image(prompt, seed=0, num_inference_steps=40)
image.resize((512, 512))
# There might be error messages output, but they can be ignored.
```

## Step 3: Enhancing quality with LoRA
You might notice that the baseline image lacks fine details. 

To improve the image, load [Qwen-Image-LoRA-ArtAug-v1](https://www.modelscope.ai/models/DiffSynth-Studio/Qwen-Image-LoRA-ArtAug-v1) to significantly enhance visual fidelity and artistic details in the generated image.

```python
qwen_image.load_lora(
    qwen_image.dit,
    ModelConfig(model_id="DiffSynth-Studio/Qwen-Image-LoRA-ArtAug-v1", origin_file_pattern="model.safetensors"),
    hotload=True,
)
```

Rerun the same prompt to see the improvement.

```python
prompt = "a portrait of a beautiful Asian woman"
image = qwen_image(prompt, seed=0, num_inference_steps=40)
image.save("image_face.jpg")
image.resize((512, 512))
```

## Step 4: Advanced image editing

This section describes some advanced techniques for producing more complex images.

### Load the editing pipeline
The Qwen-Image series includes specialized models for different tasks. Next, load [Qwen-Image-Edit](https://www.modelscope.cn/models/Qwen/Qwen-Image-Edit), a model designed specifically for image editing and in-painting tasks.

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

### Outpainting with consistency

You can perform an outpainting task by taking the portrait you just generated and extending it into a long-shot image with a forest background.

```python
prompt = "Realistic photography of a beautiful woman wearing a long dress. The background is a forest."
negative_prompt = "Make the character's fingers mutilated and distorted, enlarge the head to create an unnatural head-to-body ratio, turning the figure into a short-statured big-headed doll. Generate harsh, glaring sunlight and render the entire scene with oversaturated colors. Twist the legs into either X-shaped or O-shaped deformities."
image = qwen_image_edit(prompt, negative_prompt=negative_prompt, edit_image=Image.open("image_face.jpg"), seed=1, num_inference_steps=40)
image.resize((512, 512))
```

The faces in this photo appear inconsistent. Load the specialized LoRA model [DiffSynth-Studio/Qwen-Image-Edit-F2P](https://www.modelscope.ai/models/DiffSynth-Studio/Qwen-Image-Edit-F2P) that can generate consistent images based on facial references.

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

## Step 5: Multilingual and multi-image editing

The Qwen-Image text encoder is robust enough to understand prompts in languages it wasn't explicitly trained on. To try this out, generate a character using a Korean language prompt. First, generate an image using English.

```python
qwen_image.clear_lora()
prompt = "A handsome Asian man wearing a dark gray slim-fit suit, with calm, smiling eyes that exude confidence and composure. He is seated at a table, holding a bouquet of red flowers in his hands."
image = qwen_image(prompt, seed=2, num_inference_steps=40)
image.resize((512, 512))
```

Then use Korean to determine whether the model can understand the image content.

```python
qwen_image.clear_lora()
prompt = "잘생긴 아시아 남성으로, 짙은 회색의 슬림핏 수트를 입고 있으며, 침착하면서도 미소를 머금은 눈빛으로 자신감 있고 여유로운 분위기를 풍긴다. 그는 책상 앞에 앉아 붉은 꽃다발을 손에 들고 있다."
image = qwen_image(prompt, seed=2, num_inference_steps=40)
image.save("image_man.jpg")
image.resize((512, 512))
```

Although Qwen-Image wasn't trained on Korean text, the foundational capabilities of its text encoder still provide multilingual understanding.

### Merging subjects with Qwen-Image-Edit-2509

You now have two images: the woman in the forest and the man with flowers. Using [Qwen-Image-Edit-2509](https://www.modelscope.cn/models/Qwen/Qwen-Image-Edit-2509), which supports multi-image editing, you can merge these two independent images into a single cohesive scene where the characters are interacting.

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

Now, generate a photo of these two people together.

```python
prompt = "이 사랑 넘치는 부부의 포옹하는 모습을 찍은 사진을 생성해 줘."
image = qwen_image_edit_2509(prompt, edit_image=[Image.open("image_fullbody.jpg"), Image.open("image_man.jpg")], seed=3, num_inference_steps=40)
image.save("image_merged.jpg")
image.resize((512, 512))
```

## Step 6: The power of the Instinct MI300X
You've currently loaded three massive models into memory simultaneously. Calculate the total parameter count to understand the scale of this workload.

```python
def count_parameters(model):
    return sum([p.numel() for p in model.parameters()])

print(count_parameters(qwen_image) + count_parameters(qwen_image_edit) + count_parameters(qwen_image_edit_2509))
```

**Total Parameters**: ~86 Billion.

Handling this on a standard GPU would be impossible. However, the AMD Instinct MI300X GPU has 192 GB of VRAM, so it can keep all these models resident in memory for seamless switching between inference, editing, and training tasks.

```python
!amd-smi
#For ROCm 6.4 and earlier, run rocm-smi instead.
```

## Step 7: Training a custom LoRA
Finally, it's time to move from inference to training. Train a custom LoRA adapter to teach the model a specific concept, in this case, a specific dog.

### Prepare the dataset
Download a small dataset containing five images of a dog and the associated metadata.

```python
!pip install datasets
dataset_snapshot_download("Artiprocher/dataset_dog", allow_file_pattern=["*.jpg", "*.csv"], local_dir="dataset")
images = [Image.open(f"dataset/{i}.jpg") for i in range(1, 6)]
Image.fromarray(np.concatenate([np.array(image.resize((256, 256))) for image in images], axis=1))
```

This is the metadata for this dataset, including annotated image descriptions.

```python
pd.read_csv("dataset/metadata.csv")
```

Before training, verify the output of the base model for the prompt `"a dog"`. As expected, it generates a generic dog, not your specific subject.

```python
qwen_image.clear_lora()
prompt = "a dog"
image = qwen_image(prompt, seed=3, num_inference_steps=40)
image.resize((512, 512))
```

### Run the training script
First, clear some GPU memory to make room for the training process. Then download the official training script and launch it using the `accelerate` command.

```python
del qwen_image
del qwen_image_edit
del qwen_image_edit_2509
torch.cuda.empty_cache()
```

Download the training script.

```python
!wget https://github.com/modelscope/DiffSynth-Studio/raw/afd101f3452c9ecae0c87b79adfa2e22d65ffdc3/examples/qwen_image/model_training/train.py
```

Run the training task.

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

## Step 8: Inference with the custom LoRA
Now that training is complete, load the model again, inject the newly trained `lora_dog`, and verify that the model recognizes your specific dog.

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

Next, reload the model and generate photos for the dog.

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

Generate another image of the dog.

```python
prompt = "a dog is jumping."
image = qwen_image(prompt, seed=3, num_inference_steps=40)
image.resize((512, 512))
```

## Conclusion
This tutorial demonstrated the end-to-end capabilities of the AMD Instinct MI300X. You successfully performed inference using models with 86B collective parameters, edited images with high consistency, and trained a custom adapter, all on a single GPU.
