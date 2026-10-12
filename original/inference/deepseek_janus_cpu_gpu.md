> [Upstream source](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/inference/deepseek_janus_cpu_gpu.ipynb) · Snapshot: `0071bfd4bd3bd9525405c09a6ae69c983b09312a`

# Janus Pro DeepSeek model on AMD hardware

**Author**: Shailen Sobhee  
**Knowledge level**: Beginner

This tutorial demonstrates how to perform multimodal inference with the Janus Pro autoregressive framework, which is part of the [Janus-Series](https://github.com/deepseek-ai/Janus#-janus-series-unified-multimodal-understanding-and-generation-models) from DeepSeek AI. You'll run the model on high-performance AMD hardware, including EPYC™ CPUs and Instinct™ GPUs.

The term multimodal means that the model can understand and process information from multiple sources simultaneously, such as text and images. By unifying these different data types, known as modalities, Janus enables sophisticated understanding and generation tasks.

The tutorial also explains how to leverage the [AMD ZenDNN](https://www.amd.com/en/developer/zendnn.html) plugin (also known as zentorch) for PyTorch when executing the model on a CPU to accelerate inferencing.

## Prerequisites

Use the following setup to run this tutorial.

### Hardware

For this tutorial, you'll need a system with an AMD Instinct GPU. To run the model on the CPU and use AMD ZenDNN, you need an AMD EPYC CPU. 

This tutorial was tested on the following hardware:
* AMD Instinct MI100
* AMD Instinct MI210
* AMD Instinct MI300X
* 4th generation AMD EPYC (Genoa)
* 5th generation AMD EPYC (Turin)

### Software

* **Ubuntu 22.04**: Ensure your system is running Ubuntu 22.04 or later.
* **ROCm 6.3**: This is only required for GPU execution. Install and verify ROCm by following the [ROCm install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html).
* **PyTorch 2.6** (or later)
* **zentorch 5.0.2** (or later)
* Clone the official [Janus-Pro DeepSeek repository](https://github.com/deepseek-ai/Janus).
  
**Note**: This tutorial was tested with `torch2.7.1+rocm6.3`, `torch2.6.0+cpu`, and `zentorch-5.0.2`.

### Install and launch Jupyter Notebooks
If Jupyter is not already installed on your system, install it and launch JupyterLab using the following commands:

```
pip install jupyter
```

To start the Jupyter server, run the following command:

```
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**Note**: Ensure port `8888` is not already in use on your system before running the above command. If it is, you can specify a different port by replacing `--port=8888` with another port number, for example, `--port=8890`.

After the command executes, the terminal output displays a URL and token. Copy and paste this URL into your web browser on the host machine to access JupyterLab. After launching JupyterLab, upload this notebook to the environment and continue to follow the steps in this tutorial.

## Prepare the environment and install dependencies

The following commands install all dependencies required to successfully run this tutorial, along with the `janus` module from the DeepSeek AI GitHub repository.

```python
!pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.3
!pip install "transformers<4.50" ipywidgets
!pip install git+https://github.com/deepseek-ai/Janus.git
```

Next, run a quick check for the PyTorch environment. Validate that the GPU hardware is accessible using the ROCm backend. If not, then execute the model on the CPU.

```python
import torch 
print(f"PyTorch Version: {torch.__version__}")

print("--- GPU Verification ---")
if torch.cuda.is_available():
    print("✅ PyTorch has access to the GPU.")
    print(f"ROCm Version (detected by PyTorch): {torch.version.hip}")
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
    print(f"GPU installed on the system: {torch.cuda.get_device_name(0)}")
else:
    print("❌ PyTorch CANNOT access the GPU. Please check your ROCm installation and drivers or proceed to continue with executing on CPU")
```

Start by importing the required Python libraries required for the tutorial.

```python
import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
import time
```

Initialize the following variables for the upcoming inference process:

```python
iteration = 1
warmup = 0
max_new_tokens = 512
dtype = "bfloat16"
```

## Pipeline summary

- Choose the hardware backend: CPU or GPU
- Load the Janus-Pro model and processor.
- Define the image and question for multimodal understanding.
- Preprocess the text and image inputs.
- Generate image embeddings for the model.
- Leverage zentorch.
- Run warmup iterations to stabilize performance.
- Perform timed inference to measure latency.
- Compute and display the average generation time.
- Decode and display the AI-generated response.

## Choose the hardware backend: CPU or GPU

This example demonstrates two possible hardware backends where you can execute your AI workload: the CPU or GPU. To deploy your workload on a GPU, set `device = "cuda"`. Otherwise, to deploy on the CPU, set `device = "cpu"`. The CPU supports multiple software backends, for example, `zentorch`, the Intel® Extension for PyTorch (`ipex`), and the default PyTorch CPU backend (also known as `inductor`). You can also run the tutorial in native mode (for instance, eager-mode as opposed to graph-mode).

**Note**: zentorch plugin version 5.0.2 requires the CPU-only version of PyTorch 2.6. This is a PyTorch limitation. As a workaround, remove any previous PyTorch installations and install the CPU version.

```python
device = "cuda" # Change to "cpu" to execute on CPU
backend = "zentorch" # Available CPU backends: zentorch, inductor, ipex, or native [Eager Mode]

if device == "cpu" and backend == "zentorch":
    !pip uninstall -y torch torchvision torchaudio pytorch-triton-rocm
    !pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
    !pip install zentorch #--no-cache-dir
    import torch 
    import zentorch
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Zentorch Version: {zentorch.__version__}")

amp_enabled = True if dtype != "float32" else False
amp_dtype = getattr(torch, dtype)
```

## Step 1: Model initialization and setup

Begin by specifying the model path and initializing the necessary components for processing images and text.

```python
# Specify the path to the model  
model_path = "deepseek-ai/Janus-Pro-7B"  

# Load the multimodal chat processor and tokenizer  
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)  
tokenizer = vl_chat_processor.tokenizer  

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(  
    model_path, trust_remote_code=True  
)
```

Convert the model to use `BFloat16` precision and move it to the CPU for inference.

```python
vl_gpt = vl_gpt.to(amp_dtype).to(device).eval() 
```

## Step 2: Define the image and user input

Define an image and a text-based query to analyze its content.

```python
image = "../assets/deepseek_janus_demo_small.jpg" 
```

Use the code snippet below to check your image. This also confirms that the image path is correct.

```python
from IPython.display import Image
from IPython.core.display import HTML 
Image(image)
```

Now prepare the input payload for the vision-language model. This is achieved by constructing a `conversation` list that adheres to the model's required chat template. The user message is a dictionary containing the role, the textual question embedded with an `<image_placeholder>` token, and the corresponding image object.

```python
question = "What is happening in this image?"  

conversation = [  
    {  
        "role": "<|User|>",  
        "content": f"<image_placeholder>\n{question}",  
        "images": [image],  
    },   
]
```

## Step 3: Preprocess the image and text inputs

Load the image and convert the conversation data into a suitable format for the model.

```python
# Load images and prepare inputs  
pil_images = load_pil_images(conversation)  

# Process conversation and images into model-compatible input format  
prepare_inputs = vl_chat_processor(  
    conversations=conversation, images=pil_images, force_batchify=True  
).to(vl_gpt.device)  
```

## Step 4: Generate image embeddings

Before running inference, process the image to obtain its embeddings.

```python
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)  
```

## Step 5: Leverage the AMD ZenDNN plugin for PyTorch (zentorch) 

AMD has registered zentorch as a custom backend to `torch.compile`. This backend integrates ZenDNN optimizations after AOT Autograd through a function called `optimize()`. This function operates on the FX-based graph at the Aten IR layer to produce an optimized FX-based graph as the output. For more information about the plugin and its operations, see the [ZenDNN user guide](https://docs.amd.com/r/en-US/57300-ZenDNN-user-guide/ZenDNN).

```python
if device == "cpu":
    if(backend == "zentorch"):
        print("Backend: ZenTorch")
        import zentorch
        torch._dynamo.reset()
        vl_gpt.language_model.forward = torch.compile(vl_gpt.language_model.forward, backend="zentorch")  
    
    elif(backend == "inductor"):
        print("Backend: Inductor")
        torch._dynamo.reset()
        vl_gpt.language_model.forward = torch.compile(vl_gpt.language_model.forward)  
    
    else:
        print("Running in Eager mode")
else:
    print("We are executing on GPU therefore we won't be leveraging any CPU-acceleration software like Zentorch.")
```

###  Profiler

The PyTorch Profiler helps verify the operations (ops) of `torch.compile` and assess its effectiveness in optimizing the model. It provides insights into the model's performance by tracking execution times and pinpointing areas where optimizations can be made, ensuring that `torch.compile` is working as expected.

This part can be skipped if your focus is on performance checks rather than detailed analysis.

```python
# # Start profiling
# from torch.profiler import profile, record_function, ProfilerActivity
# def trace_handler(prof):
#     # Print profiling information after each step
#     print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))

# with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=False,schedule=torch.profiler.schedule(wait=1, warmup=1, active=1),on_trace_ready=trace_handler,) as prof:
#     for i in range(3):
#     # Run the model to get the response
#         outputs = vl_gpt.language_model.generate(
#             inputs_embeds=inputs_embeds,
#             attention_mask=prepare_inputs.attention_mask,
#             pad_token_id=tokenizer.eos_token_id,
#             bos_token_id=tokenizer.bos_token_id,
#             eos_token_id=tokenizer.eos_token_id,
#             max_new_tokens=max_new_tokens,
#             do_sample=False,
#             use_cache=True,
#         )
#         prof.step()

# # To check the DataType
# for name, param in vl_gpt.named_parameters():
#     print(f"Parameter: {name}, Shape: {param.shape}, Data Type: {param.dtype}")
#     print(f"First few values: {param.flatten()[:5]}\n") 
```

## Step 6: Warm-up inference (stabilization runs)

To ensure stable performance, run a few inference cycles without measuring the time.

```python
for i in range(warmup):  
    # Generate a response without timing for warmup  
    outputs = vl_gpt.language_model.generate(  
        inputs_embeds=inputs_embeds,  
        attention_mask=prepare_inputs.attention_mask,  
        pad_token_id=tokenizer.eos_token_id,  
        bos_token_id=tokenizer.bos_token_id,  
        eos_token_id=tokenizer.eos_token_id,
        min_new_tokens = max_new_tokens,  
        max_new_tokens = max_new_tokens,  
        do_sample=False,  
        use_cache=True,  
    )  
    print(f"WARMUP:{i+1} COMPLETED!")  
```

## Step 7: Timed inference execution

Now run the actual inference while measuring the latency for performance analysis.

```python
total_time = 0.0  

for i in range(iteration):  
    tic = time.time()  # Start time  

    # Generate response from the model  
    outputs = vl_gpt.language_model.generate(  
        inputs_embeds=inputs_embeds,  
        attention_mask=prepare_inputs.attention_mask,  
        pad_token_id=tokenizer.eos_token_id,  
        bos_token_id=tokenizer.bos_token_id,  
        eos_token_id=tokenizer.eos_token_id,  
        min_new_tokens = max_new_tokens,
        max_new_tokens = max_new_tokens,  
        do_sample=False,  
        use_cache=True,  
    )  

    toc = time.time()  # End time  
    delta = toc - tic  # Compute time taken  
    total_time = total_time + delta  
```

## Step 8: Compute and display latency

Next, calculate the average latency and print the result.

```python
total_time = total_time / iteration
print( 
    f"e2e_latency (TTFT + Generation Time) for step: {total_time:.6f} sec", 
    flush=True, 
)

tps_per_step = (max_new_tokens / total_time)
print( 
    f"Throughput: {tps_per_step:.6f} tokens/sec", 
    flush=True, 
)
```

## Step 9: Decode and display the output

Finally, decode the generated token sequence into human-readable text.

```python
answer = tokenizer.decode(outputs[0].to(device).tolist(), skip_special_tokens=True)  
print(f"{prepare_inputs['sft_format'][0]}", answer)  
```

## Pipeline summary

- Choose the hardware backend: CPU or GPU  
- Load the Janus-Pro model and processor.  
- Define the image and question for multimodal understanding.  
- Preprocess the text and image inputs.  
- Generate image embeddings for the model.  
- Leverage zentorch.  
- Run warmup iterations to stabilize performance.  
- Perform timed inference to measure latency.  
- Compute and display the average generation time.  
- Decode and display the AI-generated response.
