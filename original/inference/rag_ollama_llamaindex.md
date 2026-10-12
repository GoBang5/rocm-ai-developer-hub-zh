> [Upstream source](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/inference/rag_ollama_llamaindex.ipynb) · Snapshot: `0071bfd4bd3bd9525405c09a6ae69c983b09312a`

# Constructing a RAG system using LlamaIndex and Ollama

**Author**: Alex He  
**Knowledge level**: Beginner

AMD Radeon™ GPUs are officially supported by [ROCm](https://rocm.docs.amd.com/en/latest/index.html), ensuring compatibility with industry-standard software frameworks. This Jupyter notebook leverages [Ollama](https://ollama.com/) and [LlamaIndex](https://docs.llamaindex.ai/en/stable/), powered by ROCm, to build a Retrieval-Augmented Generation (RAG) application. LlamaIndex facilitates the creation of a pipeline from reading PDFs to indexing datasets and building a query engine, while Ollama provides the backend service for large language model (LLM) inference.

## Prerequisites

This tutorial was developed and tested using the following setup:

### Hardware

* **AMD Radeon GPUs**: Ensure you are using an [AMD Radeon GPU](https://rocm.docs.amd.com/projects/radeon/en/latest/index.html) that supports ROCm. This tutorial was tested on the AMD Radeon PRO W7900.

### Software

* **ROCm 6.2**: Install ROCm by following the [Radeon GPU install guide](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/native_linux/install-radeon.html).
* **Python 3.8**: Ensure Python is installed and accessible in your environment.

### Environment

* Root or `sudo` access is required to install and configure the software.

## Install and launch Jupyter Notebooks

If Jupyter is not already installed on your system, install it and launch JupyterLab using the following commands:

``` bash
pip install jupyter
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**Note**: Ensure port `8888` is not already in use on your system before running the above command. If it is, you can specify a different port by replacing `--port=8888` with another port number, for example, `--port=8890`.

After the command executes, the terminal output displays a URL and token. Copy and paste this URL into your web browser on the host machine to access JupyterLab. After launching JupyterLab, upload this notebook to the environment and continue to follow the steps in this tutorial.

## Install Ollama

Ollama provides seamless support for AMD ROCm GPUs, offering optimized performance without further configuration. To install Ollama on Linux, use the following command:

```python
!curl -fsSL https://ollama.com/install.sh | sh
```

Start Ollama and verify it's running:

**Note**: The Ollama installation guide is available [here](https://github.com/ollama/ollama/blob/main/docs/linux.mdx).

```python
!sudo systemctl start ollama
!sudo systemctl status ollama --no-pager
```

## Download the models

Use Ollama to pull the required models for RAG:

**Important**: If the Ollama server is running as a foreground process from the previous step, you must run the rest of this notebook in a new instance.

```python
!ollama pull nomic-embed-text
!ollama pull llama3.1:8b
```

Verify the downloaded models:

```python
!ollama list llama3.1
```

```text
NAME           ID              SIZE      MODIFIED     
llama3.1:8b    42182419e950    4.7 GB    2 months ago    
```

See the [Ollama documentation](https://github.com/ollama/ollama) for more details.

**Note**: Alternative models are available to use [here](https://ollama.com/search).

## Set up a virtual environment

Install `python3-venv` if it isn't already available (e.g. `sudo apt install python3-venv`), then create a virtual environment to avoid modifying system packages:

```python
!python3 -m venv $HOME/.venv-rag-ollama
```

## Install PyTorch (optional) 

PyTorch is optional for this tutorial. This section uses PyTorch utilities for verification purposes.

```python
!. $HOME/.venv-rag-ollama/bin/activate && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

Verify the list of installed packages:

```python
!. $HOME/.venv-rag-ollama/bin/activate && pip list | grep torch
```

```text
pytorch-triton-rocm                      3.1.0
torch                                    2.5.1+rocm6.2
torchaudio                               2.5.1+rocm6.2
torchvision                              0.20.1+rocm6.2
```

Verify the GPU functionality:

```python
import subprocess, sys
venv_site_packages = subprocess.check_output(
    '. $HOME/.venv-rag-ollama/bin/activate && python3 -c "import site; print(site.getsitepackages()[0])"',
    shell=True, text=True,
).strip()
if venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

import os
import torch
# Query GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    print('Using GPU:', torch.cuda.get_device_name(0))
    print('GPU properties:', torch.cuda.get_device_properties(0))
else:
    device = torch.device("cpu")
    print('Using CPU')
```

## Install LlamaIndex and dependencies

Use the following command to install LlamaIndex and related packages:

```python
!. $HOME/.venv-rag-ollama/bin/activate && pip install llama-index llama-index-llms-ollama llama-index-embeddings-ollama llama-index-vector-stores-chroma chromadb
```

Verify the installations:

```python
!. $HOME/.venv-rag-ollama/bin/activate && pip list | grep llama-index
```

```text
llama-index                              0.12.10
llama-index-agent-openai                 0.4.1
llama-index-cli                          0.4.0
llama-index-core                         0.12.10.post1
llama-index-embeddings-ollama            0.5.0
llama-index-embeddings-openai            0.3.1
llama-index-indices-managed-llama-cloud  0.6.3
llama-index-llms-ollama                  0.5.0
llama-index-llms-openai                  0.3.13
llama-index-multi-modal-llms-openai      0.4.2
llama-index-program-openai               0.3.1
llama-index-question-gen-openai          0.3.0
llama-index-readers-file                 0.4.3
llama-index-readers-llama-parse          0.4.0
llama-index-readers-web                  0.3.3
llama-index-vector-stores-chroma         0.4.1
```

## Build the RAG pipeline

This section explains how to configure and build the RAG pipeline.

### Set up indexing and the query engine

Import the necessary libraries:

```python
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
```

### Configure embedding and LLM models

LlamaIndex implements the Ollama client interface to interact with the Ollama service. In this example, it requests both embedding and LLM services from Ollama.

```python
# Set embedding model
emb_fn="nomic-embed-text"
Settings.embed_model = OllamaEmbedding(model_name=emb_fn)

# Set ollama model
Settings.llm = Ollama(model="llama3.1:8b", request_timeout=120.0)
```

### Download data for RAG

Download a PDF (for example, the ROCm Radeon documentation) and save it to the `./data` directory:

```python
!mkdir ./data && cd ./data && wget --recursive --level=1 --content-disposition --accept=pdf -np -nH --cut-dirs=6 https://rocm.docs.amd.com/_/downloads/radeon/en/latest/pdf/ && cd ..
```

The SimpleDirectoryReader is the most commonly used data connector.
Provide it with an input directory or a list of files and it selects the best file reader based on the file extensions.

```python
documents = SimpleDirectoryReader(input_dir="./data/").load_data()

# Check the content
print(documents[10])
```

### Create a vector dataset with Chroma

[Chroma DB](https://www.trychroma.com/) is a database that stores and queries embeddings, documents, and metadata for LLM apps that integrates well with LlamaIndex. It creates the vector dataset by sourcing the PDF file.

```python
# Initialize client and save data
db = chromadb.PersistentClient(path="./chroma_db/rocm_db")
# create collection
chroma_collection = db.get_or_create_collection("rocm_db")

# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
```

```python
# Build vector index per-document
vector_index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=20)],
)
```

### Create the query engine

Next, create the query engine with a response mode. Select the response mode based on your specific needs. For detailed guidance, see the [LlamaIndex response modes documentation](https://docs.llamaindex.ai/en/v0.10.19/module_guides/deploying/query_engine/response_modes.html).

```python
# Query your data
query_engine = vector_index.as_query_engine(response_mode="refine", similarity_top_k=10)
```

## Customize the query prompts

Define task-specific prompts:

```python
# Updating Prompt for Q&A
from llama_index.core import PromptTemplate

template = (
    "You are a car product expert who is very familiar with the car user manual and provides the guide to the end user.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the information from multiple sources and not prior knowledge\n"
    "answer the question according to the index dataset.\n"
    "if the question is not related to ROCm and Radeon GPU, just say it is not related to my knowledge base.\n"
    "if you don't know the answer, just say that I don't know.\n"
    "Answers need to be precise and concise.\n"
    "if the question is in Chinese, please translate Chinese to English in advance"
    "Query: {query_str}\n"
    "Answer: "
)
qa_template = PromptTemplate(template)
query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": qa_template}
)

template = (
    "The original query is as follows: {query_str}.\n"
    "We have provided an existing answer: {existing_answer}.\n"
    "We have the opportunity to refine the existing answer (only if needed) with some more context below.\n"
    "-------------\n"
    "{context_msg}\n"
    "-------------\n"
    "Given the new context, refine the original answer to better answer the query. If the context isn't useful, return the original answer.\n"
    "if the question is 'who are you', just say I am an expert of AMD ROCm.\n"
    "Answers need to be precise and concise.\n"
    "Refined Answer: "
)

qa_template = PromptTemplate(template)

query_engine.update_prompts(
    {"response_synthesizer:refine_template": qa_template}
)
```

## Query examples

Run the following queries:

- Query 1: Briefly describe the steps to install ROCm?

```python
response = query_engine.query("Briefly describe the steps to install ROCm?")
print(response)
```

- Query 2: Which chapter is about installing PyTorch?

```python
response = query_engine.query("Which chapter is about installing PyTorch?")
print(response)
```

- Query 3: How to verify a PyTorch installation?

```python
response = query_engine.query("How to verify a PyTorch installation?")
print(response)
```

- Query 4: Could ONNX run on a Radeon GPU?

```python
response = query_engine.query("Could ONNX run on a Radeon GPU?")
print(response)
```

## Conclusion

This tutorial demonstrates how to construct a RAG pipeline using LlamaIndex and Ollama on AMD Radeon GPUs with ROCm. For further details, see the documentation for the different components.
