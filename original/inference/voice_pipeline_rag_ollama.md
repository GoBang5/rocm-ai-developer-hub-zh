> [Upstream source](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/inference/voice_pipeline_rag_ollama.ipynb) · Snapshot: `0071bfd4bd3bd9525405c09a6ae69c983b09312a`

# Building AI pipelines for voice assistants using ROCm, LlamaIndex, and RAG

**Author**: Victoria Godsoe  
**Knowledge level**: Beginner

The following notebook demonstrates how to use AMD GPUs with [LlamaIndex](https://docs.llamaindex.ai/en/stable/) and Retrieval-Augmented Generation (RAG). It takes an input audio recording, transcribes it to text, sends the transcribed text to the RAG model, and generates a response in text format, which is then converted to speech and saved as an audio file.

## Prerequisites

This tutorial was developed and tested using the following setup.

### Operating system

* **Ubuntu 22.04**: Ensure your system is running Ubuntu version 22.04.

### Hardware

* **AMD GPUs**: This tutorial was tested on an AMD Instinct™ MI300X and an AMD Radeon™ W7900. Ensure you are using an AMD GPU with ROCm support and that your system meets the [official requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

### Software

This tutorial was tested on both AMD Radeon and AMD Instinct GPUs using the following setup:
- ROCm 6.2.0
- Python 3.10
- PyTorch 2.3.0

### Objectives

After completing this tutorial, you should understand the following concepts:

* Multi-model pipeline
* LlamaIndex with ROCm on AMD GPUs

## Prepare the inference environment

To set up the inference environment, follow these steps:

1. Create a conda environment:
   ```bash
   conda create -n rocm python=3.10
   ```

1. Activate the environment:
   ```bash
   conda activate rocm
   ```

1. Install the PyTorch for ROCm software:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
   ```

1. Install Ollama (if not previously installed). This step requires `curl`:
   ```bash
   sudo apt install curl -y
   curl -fsSL https://ollama.com/install.sh | sh
   ```

1. Launch the Ollama server if it isn't already running:
   ```bash
   ollama serve &
   ```

1. Pull `llama3` with Ollama:
   ```bash
   ollama pull llama3
   ```

1. Install the example dependencies:
   ```bash
   pip install llama-index llama-index-llms-ollama llama-index-embeddings-ollama llama-index-embeddings-huggingface openai-whisper transformers ChatTTS
   ```

1. Include an audio file (for example, `summarize_question.wav`). Place it in the current working directory.

### Install and launch Jupyter

Inside the Docker container, install Jupyter using the following command:

```bash
pip install jupyter
```

Then start the Jupyter server:

```bash
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**Note**: Ensure port `8888` is not already in use on your system before running the above command. If it is, you can specify a different port by replacing `--port=8888` with another port number, for example, `--port=8890`.

## Import the packages

Import the following packages:

* **os**: Operating-system-dependent functionality.
* **whisper**: A speech-recognition library.
* **torch**: A PyTorch library for tensor computations and deep learning.
* **llama_index.core**: Core functionality for the Llama Index.
* **llama_index.embeddings.huggingface**: Support for embedding HuggingFace.
* **llama_index.llms.ollama**: Functionality for the Ollama language model.
* **ChatTTS**: A text-to-speech conversion library.
* **torchaudio**: An audio-processing library.
* **IPython.display**: For displaying audio in Jupyter notebooks.

```python
# System Imports
import os
import numpy as np
import re

# Imports for Speech to Text
import whisper
import torch

# Imports for RAG Model
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# Imports for Text to Speech
import ChatTTS
import torchaudio
from IPython.display import Audio
```

## Set up the environment

1. Optionally, set the environment variables to enable experimental features in PyTorch ROCm.
2. Verify the PyTorch version and GPU availability.
3. Select the computation device:
   - Use the GPU if available and print its properties.
   - Fall back to the CPU otherwise.

```python
# Set the environment variable for experimental features (optional)
os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '1'
os.environ['HIP_VISIBLE_DEVICES'] = "0"

print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Check GPU availability and properties
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU (no GPU detected)")
```

## Transcribe speech to text

The following section performs speech-to-text transcription using the Whisper model.

First, download the sample audio file for this tutorial.

```python
!curl -L https://raw.githubusercontent.com/ROCm/gpuaidev/main/docs/notebooks/assets/summarize_question.wav -o summarize_question.wav
```

Load the audio file and transcribe the speech content into text.

```python
AUDIO_FILE = "summarize_question.wav"
Audio(AUDIO_FILE, rate=24_000, autoplay=True)
```

Now transcribe the speech content into text.

```python
# Install ffmpeg (required by openai-whisper for audio decoding)
!apt-get update -qq && apt-get install -y -qq ffmpeg

# Speech-to-Text with Whisper
try:
    model = whisper.load_model("base")
    result = model.transcribe(AUDIO_FILE)
    input_text = result["text"]
    print(f"Transcribed text: {input_text}")
except Exception as e:
    print(f"Error in speech-to-text: {e}")
    exit(1)
```

## Add the RAG model

To use a RAG model, provide the context that you'd like the LLM to use for the queries. This example is configured to use the documents in the `data` folder. If you don't have any documents yet, you can add your own or download the one below.

```python
DATA_DIR = "./data"

# Check if the data directory exists, and create it if it doesn't
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"Data directory '{DATA_DIR}' created. Please add a file of your choosing or use the cell below to download sample text.")
    exit(1)
else:
    # Check if data directory is empty
    if not os.listdir(DATA_DIR):
        print(f"Data directory '{DATA_DIR}' is empty. Please add a file of your choosing or use the cell below to download sample text.")
        exit(1)
```

If the `data` directory is empty, run the following cell:

```python
## OPTIONAL - Run this cell if your data directory is empty
!mkdir -p data && curl -L https://www.gutenberg.org/cache/epub/11/pg11.txt -o data/pg11.txt
```

Verify the data file now exists in the `data` directory.

```python
# View the files in your data directory
print("Files in data directory:", os.listdir("data"))
documents = SimpleDirectoryReader(DATA_DIR).load_data()
```

For the embedding model, use "bge-base" from `HuggingFaceEmbedding`. Confirm that the Ollama server is running because it supplies Llama-3 for the LLM.  

Next, create a `VectorStoreIndex` from the loaded documents and initialize a query engine with the index. Then issue your query using the text output from the Whisper model. Print the response so you can compare it against the audio output in the next step.

```python
# Initialize embedding and LLM models
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

try:
    Settings.llm = Ollama(model="llama3", request_timeout=360.0)
except Exception as e:
    print(f"Error connecting to Ollama server: {e}")
    exit(1)

# Build and query the vector index
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(streaming=True, response_mode="compact", similarity_top_k=3)
response = query_engine.query(input_text)

# Function to convert StreamingResponse to string
def streaming_response_to_string(streaming_response):
    text = ""
    for chunk in streaming_response.response_gen:
        if isinstance(chunk, dict) and "text" in chunk:
            text += chunk["text"]
        else:
            text += str(chunk)
    return text

# Convert response to string
response_text = streaming_response_to_string(response)
print(f"Generated response: {response_text}")
```

## Perform text-to-speech conversion

The following example performs text-to-speech conversion using the ChatTTS library and saves the output audio to a file.

This example uses the following constants:
* `OUTPUT_AUDIO_FILE` (`str`): The name of the output audio file.
* `SAMPLE_RATE` (`int`): The sample rate for the output audio file.

It provides the following functionality:
* Initializes a `ChatTTS.Chat` object.
* Loads the chat model without compilation for faster loading. (Set `compile=True` for better performance.)
* Converts the response text from the previous step to speech.
* Saves the generated audio to the specified output file using torchaudio.

```python
OUTPUT_AUDIO_FILE = "voice_pipeline_response.wav"
SAMPLE_RATE = 24000

# Text cleanup function for TTS
def sanitize_input(text):
    sanitized_text = text.replace('-', '')  # Remove hyphens
    sanitized_text = sanitized_text.replace('(', '').replace(')', '')  # Remove parentheses
    return sanitized_text.strip()

# Text-to-Speech processing
try:
    sanitized_response = re.sub(r"[^a-zA-Z0-9.,?! ]", "", response_text)  # Remove special characters
    print(f"Sanitized response for TTS: {sanitized_response}")
    sanitized_response = [sanitized_response]

    chat = ChatTTS.Chat()
    chat.load(compile=False) # Set to True for better performance

    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb = chat.sample_random_speaker(),
    )

    wavs = chat.infer(
        sanitized_response,
        params_infer_code=params_infer_code,
    )
    try:
        torchaudio.save(OUTPUT_AUDIO_FILE, torch.from_numpy(wavs[0]).unsqueeze(0), SAMPLE_RATE)
    except:
        torchaudio.save(OUTPUT_AUDIO_FILE, torch.from_numpy(wavs[0]), SAMPLE_RATE)

except Exception as e:
    print(f"Error in text-to-speech: {e}")
    exit(1)

finally:
    if 'chat' in locals():
        chat.unload()
```

Play the following cell to hear the generated speech.

```python
Audio(wavs[0], rate=24_000, autoplay=True)
```
