> [Upstream source](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/pretrain/ddim_pretrain.ipynb) · Snapshot: `0071bfd4bd3bd9525405c09a6ae69c983b09312a`

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

```python
!pip install torch torchvision  --index-url https://download.pytorch.org/whl/rocm6.2
!pip install matplotlib transformers diffusers datasets
```

Verify the Torch installation:

```python
import torch

print("Torch Version:", torch.__version__)
print("Is ROCm available:", torch.cuda.is_available())
```

### Creating the Config class

Next, define the `Config` class to store the training parameters. This lets you easily reuse and modify the configuration for future runs. Adjusting these parameters affects training times and the quality of the generated images, so experiment with different values to find the optimal setup.

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

### Creating the training dataset

Load the dataset and display a few images to confirm that everything is ready for training.

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

### Creating the transforms for the dataset

Before training, apply transformations to the images to ensure they have the correct size and format. This involves resizing the images, normalizing their pixel values, and applying data augmentation.

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

### Creating the model 

To create the model, use a [U-Net](https://arxiv.org/abs/1505.04597) architecture, implemented using the `UNet2DModel` from the `diffusers` library. U-Net is a widely used architecture for denoising diffusion models due to its encoder-decoder structure with skip connections, which help preserve spatial information while allowing deep feature extraction. This makes it highly effective for generating high-quality images in diffusion-based models.

`AttnDownBlock2D` and `AttnUpBlock2D` improve the model’s ability to capture long-range dependencies, which standard convolutions struggle with. By letting each pixel attend to relevant regions across the image, attention enhances feature refinement and structure preservation, leading to more coherent and detailed image generation.

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

With the model defined, you can now simulate the diffusion process using a noise scheduler. This adds noise to an image in a step-by-step manner, which the model later learns to reverse during training.

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

## Setting up the optimizer and learning rate scheduler

Use the Adam optimizer, which is one of the most widely used optimizers in deep learning due to its adaptive learning rate and efficient performance across various tasks.

For learning rate scheduling, apply a "cosine decay schedule" with warmup, as implemented in the `diffusers` library. This approach gradually increases the learning rate at the start of training (the warmup phase) to stabilize updates, then decays it smoothly following a cosine curve. This helps prevent sudden drops in performance and lets the model converge more effectively.

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

### Evaluation function

This function generates images using the trained model and saves them as a grid for visual inspection. Use the DDIMPipeline from the `diffusers` library to create the images and save the output in the `samples` directory.

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

### Training loop

The following code defines the training loop for the DDIM model. It ensures stable and efficient training by iterating over the dataset and updating the parameters for the model at each epoch. At regular intervals, it performs evaluations and saves model checkpoints to monitor the progress and improve performance.

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

### Launching the training

Use the following command to launch the training process:

```python
train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
```

### Visualizing the generated images

This section demonstrates how to visualize the images generated during the training process. This helps you inspect the quality and diversity of the generated images.

```python
import glob
sample_images = sorted(glob.glob(f"{config.output_dir}/samples/*.png"))
Image.open(sample_images[-1])
```

## References


[https://huggingface.co/docs/diffusers/en/tutorials/basic_training](https://huggingface.co/docs/diffusers/en/tutorials/basic_training)

 This tutorial provides a comprehensive guide on how to train models using the `diffusers` library, from setting up the dataset to optimizing and running training loops.

---

[https://huggingface.co/docs/diffusers/en/tutorials/autopipeline](https://huggingface.co/docs/diffusers/en/tutorials/autopipeline)

 Learn how to leverage the Hugging Face AutoPipeline feature, which simplifies the training pipeline by automatically managing different components.

---

[https://huggingface.co/docs/diffusers/en/using-diffusers/write_own_pipeline](https://huggingface.co/docs/diffusers/en/using-diffusers/write_own_pipeline)

 A step-by-step guide on how to build your own custom pipeline using the `diffusers` library, offering flexibility to adapt models to specific use cases.
