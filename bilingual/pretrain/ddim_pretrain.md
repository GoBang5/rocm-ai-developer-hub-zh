> 原文作者：Cătălin (Constantin) Milu  
> [AMD 官方教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/ddim_pretrain.html) · [固定版本源码](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/pretrain/ddim_pretrain.ipynb)  
> 上游版本：`0071bfd4bd3bd9525405c09a6ae69c983b09312a` · 同步日期：2026-10-12  
> 翻译状态：机器初译，待逐篇人工校对；代码与命令保持原文，未在 AMD GPU 上运行验证。

<!-- cell:1 -->

<a id="custom-diffusion-model-with-pytorch"></a>

# 使用 PyTorch 训练自定义扩散模型
**作者**：Cătălin (Constantin) Milu
**知识水平**：初学者

本教程将引导您了解如何在 AMD GPU 上使用 [Hugging Face Diffusers 库](https://github.com/huggingface/diffusers) 预训练 **去噪扩散隐式模型 (DDIM)**。您将训练基于 U-Net 的 DDIM 模型，以从 Flowers-102 数据集生成逼真的花朵图像。

<a id="model-and-dataset-overview"></a>

## 模型和数据集概述

本教程使用 Flowers-102 数据集，其中包含 102 个不同类别的花朵图像。该数据集提供了各种纹理、颜色和形状，使其成为训练扩散模型的理想选择。在 预训练 期间，模型学习生成不同的花卉图像。然后，微调 调整模型以生成更高质量的图像和特定风格的输出。

<a id="prerequisites"></a>

## 前置条件

本教程是使用以下设置进行开发和测试的：

<a id="operating-system"></a>

### 操作系统

* **Ubuntu 22.04**

<a id="hardware"></a>

### 硬件

* **AMD Instinct™ GPU**：本教程在 AMD Instinct MI300X GPU 上进行了测试。确保您使用支持 ROCm 的 AMD Instinct GPU，并且您的系统满足[官方要求](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)。

<a id="software"></a>

### 软件

* **ROCm 6.2.4**：按照 [ROCm 安装指南](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html) 安装并验证 ROCm。
**Python 3.7+**

<a id="prepare-the-inference-environment"></a>

## 准备推理环境

本节首先创建一个虚拟环境，然后启动Jupyter服务器。 

<a id="install-the-dependencies"></a>

### 安装依赖项

首先创建一个虚拟环境：

```
python3 -m venv venv
source venv/bin/activate 
```

<a id="install-and-launch-jupyter"></a>

### 安装并启动 Jupyter

使用以下命令安装 Jupyter：

```bash
pip install jupyter
```

然后启动Jupyter服务器：

```bash
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**注意**：在运行上述命令之前，请确保您的系统上尚未使用端口 `8888`。如果是，您可以通过将 `--port=8888` 替换为其他端口号来指定其他端口，例如 `--port=8890`。


安装依赖项：

<details markdown="1">
<summary>查看英文原文 · 单元 1</summary>

# Custom diffusion model with PyTorch

**Author**: Cătălin (Constantin) Milu  
**Knowledge level**: Beginner

This tutorial walks you through how to pretrain a **Denoising Diffusion Implicit Model (DDIM)** using the [Hugging Face Diffusers library](https://github.com/huggingface/diffusers) on AMD GPUs. You'll train a U-Net-based DDIM model to generate realistic flower images from the Flowers-102 dataset.

## Model and dataset overview

This tutorial uses the Flowers-102 dataset, which contains images of flowers from 102 different categories. The dataset provides a wide variety of textures, colors, and shapes, making it ideal for training a diffusion model. During pretraining, the model learns to generate diverse flower images. Fine-tuning then adapts the model to generate higher-quality images and style-specific outputs.

## Prerequisites

This tutorial was developed and tested using the following setup:

### Operating system

* **Ubuntu 22.04**

### Hardware

* **AMD Instinct™ GPUs**:  This tutorial was tested on an AMD Instinct MI300X GPU. Ensure you use an AMD Instinct GPU with ROCm support and that your system meets the [official requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

### Software

* **ROCm 6.2.4**: Install and verify ROCm by following the [ROCm install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html).
* **Python 3.7+**

## Prepare the inference environment

This section first creates a virtual environment and then starts the Jupyter server. 

### Install the dependencies

Start by creating a virtual environment:

```
python3 -m venv venv
source venv/bin/activate 
```

### Install and launch Jupyter

Install Jupyter using the following command:

```bash
pip install jupyter
```

Then start the Jupyter server:

```bash
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**Note**: Ensure port `8888` is not already in use on your system before running the above command. If it is, you can specify a different port by replacing `--port=8888` with another port number, for example, `--port=8890`.


Install the dependencies:

</details>

```python
!pip install torch torchvision  --index-url https://download.pytorch.org/whl/rocm6.2
!pip install matplotlib transformers diffusers datasets
```

<!-- cell:3 -->

验证 Torch 安装：

<details markdown="1">
<summary>查看英文原文 · 单元 3</summary>

Verify the Torch installation:

</details>

```python
import torch

print("Torch Version:", torch.__version__)
print("Is ROCm available:", torch.cuda.is_available())
```

<!-- cell:5 -->

<a id="creating-the-config-class"></a>

### 创建 Config 类

接下来，定义 `Config` 类来存储训练参数。这使您可以轻松地重用和修改配置以供将来运行。调整这些参数会影响训练时间和生成图像的质量，因此请尝试不同的值以找到最佳设置。

<details markdown="1">
<summary>查看英文原文 · 单元 5</summary>

### Creating the Config class

Next, define the `Config` class to store the training parameters. This lets you easily reuse and modify the configuration for future runs. Adjusting these parameters affects training times and the quality of the generated images, so experiment with different values to find the optimal setup.

</details>

```python
import torch
from dataclasses import dataclass

@dataclass
class Config:
    image_size = 128                                # Size of the training images
    train_batch_size = 16                           # Batch size for training
    eval_batch_size = 16                            # Batch size for evaluation
    num_epochs = 100                                # Total number of training epochs
    learning_rate = 1e-4                            # Learning rate for optimization
    lr_warmup_steps = 500                           # Warmup steps for learning rate scheduling
    save_image_epochs = 10                          # Frequency of saving generated images
    save_model_epochs = 30                          # Frequency of saving model checkpoints
    output_dir = "ddim-flowers-128"                 # Output directory for model and images
    seed = 36                                       # Random seed for reproducibility
    dataset_name = "huggan/flowers-102-categories"  # Name of the dataset
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
config = Config()
```

<!-- cell:7 -->

<a id="creating-the-training-dataset"></a>

### 创建训练数据集

加载数据集并显示一些图像以确认一切都已准备好进行训练。

<details markdown="1">
<summary>查看英文原文 · 单元 7</summary>

### Creating the training dataset

Load the dataset and display a few images to confirm that everything is ready for training.

</details>

```python
from datasets import load_dataset
import matplotlib.pyplot as plt

# Load dataset from Hugging Face
dataset = load_dataset(config.dataset_name, split="train")

# Visualize some images
fig, axs = plt.subplots(1, 5, figsize=(16, 4))
for i, image in enumerate(dataset["image"]):
    axs[i].imshow(image)
    axs[i].axis("off")  
    if (i + 1) % 5 == 0:  # Show only 5 images
        break
plt.show()
```

<!-- cell:9 -->

<a id="creating-the-transforms-for-the-dataset"></a>

### 创建数据集的转换

在训练之前，对图像应用转换以确保它们具有正确的大小和格式。这涉及调整图像大小、标准化其像素值以及应用数据增强。

<details markdown="1">
<summary>查看英文原文 · 单元 9</summary>

### Creating the transforms for the dataset

Before training, apply transformations to the images to ensure they have the correct size and format. This involves resizing the images, normalizing their pixel values, and applying data augmentation.

</details>

```python
from torchvision import transforms

transformations = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1] range
    ]
)

def transform(examples):
    images = [transformations(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}

dataset.set_transform(transform)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
```

<!-- cell:11 -->

<a id="creating-the-model"></a>

### 创建模型 

要创建模型，请使用 [U-Net](https://arxiv.org/abs/1505.04597) 架构，该架构使用 `diffusers` 库中的 `UNet2DModel` 实现。 U-Net 是一种广泛使用的去噪扩散模型架构，因为它的编码器-解码器结构具有跳跃连接，有助于保留空间信息，同时允许深度特征提取。这使得它能够非常有效地在基于扩散的模型中生成高质量图像。

`AttnDownBlock2D` 和 `AttnUpBlock2D` 提高了模型捕获远程依赖关系的能力，而标准卷积则难以解决这一问题。通过让每个像素关注图像中的相关区域，注意力增强了特征细化和结构保留，从而生成更加连贯和详细的图像。

<details markdown="1">
<summary>查看英文原文 · 单元 11</summary>

### Creating the model 

To create the model, use a [U-Net](https://arxiv.org/abs/1505.04597) architecture, implemented using the `UNet2DModel` from the `diffusers` library. U-Net is a widely used architecture for denoising diffusion models due to its encoder-decoder structure with skip connections, which help preserve spatial information while allowing deep feature extraction. This makes it highly effective for generating high-quality images in diffusion-based models.

`AttnDownBlock2D` and `AttnUpBlock2D` improve the model’s ability to capture long-range dependencies, which standard convolutions struggle with. By letting each pixel attend to relevant regions across the image, attention enhances feature refinement and structure preservation, leading to more coherent and detailed image generation.

</details>

```python
from diffusers import UNet2DModel

model = UNet2DModel(
    sample_size=config.image_size, 
    in_channels=3,  
    out_channels=3,  
    layers_per_block=2, 
    dropout=0.1, 
    block_out_channels=(128, 128, 256, 256, 512, 512),  # Channels per block
    down_block_types=(
        "DownBlock2D",  
        "DownBlock2D",
        "AttnDownBlock2D",  
        "DownBlock2D",
        "AttnDownBlock2D",  
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  
        "AttnUpBlock2D", 
        "UpBlock2D",
        "AttnUpBlock2D", 
        "UpBlock2D",
        "UpBlock2D",
    ),
).to(config.device)

# Printing the model summary 

total_params = sum(param.numel() for param in model.parameters())
total_size_mb = total_params * 4 / (1024 ** 2)

print("Total Model Parameters:", f"{total_params:,}")
print("Total Model Size: {:.2f} MB".format(total_size_mb))
```

<!-- cell:13 -->

定义模型后，您现在可以使用噪声调度程序来模拟扩散过程。这会逐步向图像添加噪声，模型稍后会在训练过程中学习将其逆转。

<details markdown="1">
<summary>查看英文原文 · 单元 13</summary>

With the model defined, you can now simulate the diffusion process using a noise scheduler. This adds noise to an image in a step-by-step manner, which the model later learns to reverse during training.

</details>

```python
from PIL import Image
from diffusers import DDIMScheduler

# Select an image from the dataset
image = dataset[0]["images"].unsqueeze(0)

# Define the noise scheduler
noise_scheduler = DDIMScheduler(num_train_timesteps=1000)

# Generate random noise
noise = torch.randn_like(image)
timesteps = torch.tensor([100], dtype=torch.long)

# Add noise to the image
noisy_image = noise_scheduler.add_noise(image, noise, timesteps)

# Convert to PIL image for visualization
Image.fromarray(
    ((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).clamp(0, 255).byte().numpy()[0]
)
```

<!-- cell:15 -->

<a id="setting-up-the-optimizer-and-learning-rate-scheduler"></a>

## 设置优化器和 学习率 调度程序

使用 Adam 优化器，它是深度学习中使用最广泛的优化器之一，因为它具有自适应 学习率 和跨各种任务的高效性能。

对于 学习率 调度，应用带预热的“余弦衰减调度”，如 `diffusers` 库中实现的那样。这种方法在训练开始时（预热阶段）逐渐增加 学习率 以稳定更新，然后按照余弦曲线平滑衰减。这有助于防止性能突然下降，并使模型更有效地收敛。

<details markdown="1">
<summary>查看英文原文 · 单元 15</summary>

## Setting up the optimizer and learning rate scheduler

Use the Adam optimizer, which is one of the most widely used optimizers in deep learning due to its adaptive learning rate and efficient performance across various tasks.

For learning rate scheduling, apply a "cosine decay schedule" with warmup, as implemented in the `diffusers` library. This approach gradually increases the learning rate at the start of training (the warmup phase) to stabilize updates, then decays it smoothly following a cosine curve. This helps prevent sudden drops in performance and lets the model converge more effectively.

</details>

```python
from diffusers.optimization import get_cosine_schedule_with_warmup

# Initialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# Initialize the learning rate scheduler
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)
```

<!-- cell:17 -->

<a id="evaluation-function"></a>

### 评估函数

该函数使用经过训练的模型生成图像，并将其保存为网格以供目视检查。使用 `diffusers` 库中的 DDIMPipeline 创建图像并将输出保存在 `samples` 目录中。

<details markdown="1">
<summary>查看英文原文 · 单元 17</summary>

### Evaluation function

This function generates images using the trained model and saves them as a grid for visual inspection. Use the DDIMPipeline from the `diffusers` library to create the images and save the output in the `samples` directory.

</details>

```python
from diffusers import DDIMPipeline
from diffusers.utils import make_image_grid
import os

def evaluate(config, epoch, pipeline):
    # Generate images with the trained pipeline
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.Generator(device="cpu").manual_seed(config.seed)
    ).images
    
    # Create a grid from the generated images
    image_grid = make_image_grid(images, rows=4, cols=4)
    
    # Define the directory to save the generated images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    
    # Save the image grid
    image_grid.save(f"{test_dir}/{epoch}.png")
```

<!-- cell:19 -->

<a id="training-loop"></a>

### 训练循环

以下代码定义了 DDIM 模型的训练循环。它通过迭代数据集并在每个时期更新模型参数来确保稳定和高效的训练。它定期执行评估并保存模型检查点以监控进度并提高性能。

<details markdown="1">
<summary>查看英文原文 · 单元 19</summary>

### Training loop

The following code defines the training loop for the DDIM model. It ensures stable and efficient training by iterating over the dataset and updating the parameters for the model at each epoch. At regular intervals, it performs evaluations and saves model checkpoints to monitor the progress and improve performance.

</details>

```python
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Ensure output directory is set up
    os.makedirs(config.output_dir, exist_ok=True)

    global_step = 0
    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")

        # Iterate through the batches in the dataloader
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"].to(config.device)
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Generate random timesteps for the noise scheduler
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=clean_images.device,
                dtype=torch.int64
            )

            # Add noise to the clean images
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            # Forward pass to predict the noise
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)
                
            # Backpropagate the loss
            loss.backward()
                
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()        # Update model weights
            lr_scheduler.step()     # Update the learning rate
            optimizer.zero_grad()   # Reset gradients

            global_step += 1
            epoch_loss += loss.detach().item()
            
            # Logging and progress tracking
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            progress_bar.set_postfix(**logs)
            progress_bar.update(1)

        # Print average loss for the epoch
        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
        
        
        pipeline = DDIMPipeline(unet=model, scheduler=noise_scheduler)
        
        # Save images at regular intervals
        if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
            evaluate(config, epoch, pipeline)
        
        # Save the model at regular intervals
        if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
            pipeline.save_pretrained(config.output_dir)            
```

<!-- cell:21 -->

<a id="launching-the-training"></a>

### 启动培训

使用以下命令启动训练过程：

<details markdown="1">
<summary>查看英文原文 · 单元 21</summary>

### Launching the training

Use the following command to launch the training process:

</details>

```python
train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
```

<!-- cell:23 -->

<a id="visualizing-the-generated-images"></a>

### 可视化生成的图像

本节演示如何可视化训练过程中生成的图像。这可以帮助您检查生成图像的质量和多样性。

<details markdown="1">
<summary>查看英文原文 · 单元 23</summary>

### Visualizing the generated images

This section demonstrates how to visualize the images generated during the training process. This helps you inspect the quality and diversity of the generated images.

</details>

```python
import glob
sample_images = sorted(glob.glob(f"{config.output_dir}/samples/*.png"))
Image.open(sample_images[-1])
```

<!-- cell:25 -->

<a id="references"></a>

## 参考文献


[https://huggingface.co/docs/diffusers/en/tutorials/basic_training](https://huggingface.co/docs/diffusers/en/tutorials/basic_training)

 本教程提供了有关如何使用 `diffusers` 库训练模型的全面指南，从设置数据集到优化和运行训练循环。

---

[https://huggingface.co/docs/diffusers/en/tutorials/autopipeline](https://huggingface.co/docs/diffusers/en/tutorials/autopipeline)

 了解如何利用 Hugging Face AutoPipeline 功能，该功能通过自动管理不同的组件来简化训练流程。

---

[https://huggingface.co/docs/diffusers/en/using-diffusers/write_own_pipeline](https://huggingface.co/docs/diffusers/en/using-diffusers/write_own_pipeline)

 有关如何使用 `diffusers` 库构建您自己的自定义管道的分步指南，提供使模型适应特定用例的灵活性。

<details markdown="1">
<summary>查看英文原文 · 单元 25</summary>

## References


[https://huggingface.co/docs/diffusers/en/tutorials/basic_training](https://huggingface.co/docs/diffusers/en/tutorials/basic_training)

 This tutorial provides a comprehensive guide on how to train models using the `diffusers` library, from setting up the dataset to optimizing and running training loops.

---

[https://huggingface.co/docs/diffusers/en/tutorials/autopipeline](https://huggingface.co/docs/diffusers/en/tutorials/autopipeline)

 Learn how to leverage the Hugging Face AutoPipeline feature, which simplifies the training pipeline by automatically managing different components.

---

[https://huggingface.co/docs/diffusers/en/using-diffusers/write_own_pipeline](https://huggingface.co/docs/diffusers/en/using-diffusers/write_own_pipeline)

 A step-by-step guide on how to build your own custom pipeline using the `diffusers` library, offering flexibility to adapt models to specific use cases.

</details>
