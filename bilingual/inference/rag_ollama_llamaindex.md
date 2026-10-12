> 原文作者：Alex He  
> [AMD 官方教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/rag_ollama_llamaindex.html) · [固定版本源码](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/inference/rag_ollama_llamaindex.ipynb)  
> 上游版本：`0071bfd4bd3bd9525405c09a6ae69c983b09312a` · 同步日期：2026-10-12  
> 翻译状态：机器初译，待逐篇人工校对；代码与命令保持原文，未在 AMD GPU 上运行验证。

<!-- cell:1 -->

<a id="constructing-a-rag-system-using-llamaindex-and-ollama"></a>

# 使用 LlamaIndex 和 Ollama 构建 RAG
**作者**：Alex He
**知识水平**：初学者

<details markdown="1">
<summary>查看英文原文 · 单元 1</summary>

# Constructing a RAG system using LlamaIndex and Ollama

**Author**: Alex He  
**Knowledge level**: Beginner

</details>

<!-- cell:2 -->

AMD Radeon™ GPU 得到 [ROCm](https://rocm.docs.amd.com/en/latest/index.html) 的正式支持，确保与行业标准软件框架的兼容性。此 Jupyter Notebook利用由 ROCm 提供支持的 [Ollama](https://ollama.com/) 和 [LlamaIndex](https://docs.llamaindex.ai/en/stable/) 来构建 检索增强生成 (RAG) 应用程序。 LlamaIndex 有助于创建从读取 PDF 到索引数据集和构建查询引擎的管道，而 Ollama 为 大语言模型 (LLM) 推理 提供后端服务。

<details markdown="1">
<summary>查看英文原文 · 单元 2</summary>

AMD Radeon™ GPUs are officially supported by [ROCm](https://rocm.docs.amd.com/en/latest/index.html), ensuring compatibility with industry-standard software frameworks. This Jupyter notebook leverages [Ollama](https://ollama.com/) and [LlamaIndex](https://docs.llamaindex.ai/en/stable/), powered by ROCm, to build a Retrieval-Augmented Generation (RAG) application. LlamaIndex facilitates the creation of a pipeline from reading PDFs to indexing datasets and building a query engine, while Ollama provides the backend service for large language model (LLM) inference.

</details>

<!-- cell:3 -->

<a id="prerequisites"></a>

## 前置条件

本教程是使用以下设置进行开发和测试的：

<a id="hardware"></a>

### 硬件

* **AMD Radeon GPU**：确保您使用的是支持 ROCm 的 [AMD Radeon GPU](https://rocm.docs.amd.com/projects/radeon/en/latest/index.html)。本教程在 AMD Radeon PRO W7900 上进行了测试。

<a id="software"></a>

### 软件

* **ROCm 6.2**：按照 [Radeon GPU 安装指南](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/native_linux/install-radeon.html) 安装 ROCm。
* **Python 3.8**：确保 Python 已安装并可在您的环境中访问。

<a id="environment"></a>

### 环境

* 安装和配置软件需要 root 或 `sudo` 访问权限。

<details markdown="1">
<summary>查看英文原文 · 单元 3</summary>

## Prerequisites

This tutorial was developed and tested using the following setup:

### Hardware

* **AMD Radeon GPUs**: Ensure you are using an [AMD Radeon GPU](https://rocm.docs.amd.com/projects/radeon/en/latest/index.html) that supports ROCm. This tutorial was tested on the AMD Radeon PRO W7900.

### Software

* **ROCm 6.2**: Install ROCm by following the [Radeon GPU install guide](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/native_linux/install-radeon.html).
* **Python 3.8**: Ensure Python is installed and accessible in your environment.

### Environment

* Root or `sudo` access is required to install and configure the software.

</details>

<!-- cell:4 -->

<a id="install-and-launch-jupyter-notebooks"></a>

## 安装并启动 Jupyter Notebooks

如果您的系统上尚未安装 Jupyter，请安装它并使用以下命令启动 JupyterLab：

``` bash
pip install jupyter
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**注意**：在运行上述命令之前，请确保您的系统上尚未使用端口 `8888`。如果是，您可以通过将 `--port=8888` 替换为其他端口号来指定其他端口，例如 `--port=8890`。

命令执行后，终端输出将显示 URL 和令牌。复制此 URL 并将其粘贴到主机上的 Web 浏览器中以访问 JupyterLab。启动 JupyterLab 后，将此Notebook上传到环境并继续按照本教程中的步骤进行操作。

<a id="install-ollama"></a>

## 安装奥拉玛

Ollama 为 AMD ROCm GPU 提供无缝支持，无需进一步配置即可提供优化的性能。要在 Linux 上安装 Ollama，请使用以下命令：

<details markdown="1">
<summary>查看英文原文 · 单元 4</summary>

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

</details>

```python
!curl -fsSL https://ollama.com/install.sh | sh
```

<!-- cell:6 -->

启动 Ollama 并验证其正在运行：

**注意**：Ollama 安装指南可在[此处](https://github.com/ollama/ollama/blob/main/docs/linux.mdx) 获取。

<details markdown="1">
<summary>查看英文原文 · 单元 6</summary>

Start Ollama and verify it's running:

**Note**: The Ollama installation guide is available [here](https://github.com/ollama/ollama/blob/main/docs/linux.mdx).

</details>

```python
!sudo systemctl start ollama
!sudo systemctl status ollama --no-pager
```

<!-- cell:8 -->

<a id="download-the-models"></a>

## 下载模型

使用 Ollama 拉取 RAG 所需的模型：

**重要**：如果 Ollama 服务器作为上一步中的前台进程运行，则必须在新实例中运行此Notebook的其余部分。

<details markdown="1">
<summary>查看英文原文 · 单元 8</summary>

## Download the models

Use Ollama to pull the required models for RAG:

**Important**: If the Ollama server is running as a foreground process from the previous step, you must run the rest of this notebook in a new instance.

</details>

```python
!ollama pull nomic-embed-text
!ollama pull llama3.1:8b
```

<!-- cell:10 -->

验证下载的模型：

<details markdown="1">
<summary>查看英文原文 · 单元 10</summary>

Verify the downloaded models:

</details>

```python
!ollama list llama3.1
```

```text
NAME           ID              SIZE      MODIFIED     
llama3.1:8b    42182419e950    4.7 GB    2 months ago    
```

<!-- cell:12 -->

有关更多详细信息，请参阅 [Ollama 文档](https://github.com/ollama/ollama)。

**注意**：[此处](https://ollama.com/search) 可以使用替代型号。

<details markdown="1">
<summary>查看英文原文 · 单元 12</summary>

See the [Ollama documentation](https://github.com/ollama/ollama) for more details.

**Note**: Alternative models are available to use [here](https://ollama.com/search).

</details>

<!-- cell:13 -->

<a id="set-up-a-virtual-environment"></a>

## 设置虚拟环境

如果 `python3-venv` 尚不可用，请安装它（例如 `sudo apt install python3-venv`），然后创建一个虚拟环境以避免修改系统包：

<details markdown="1">
<summary>查看英文原文 · 单元 13</summary>

## Set up a virtual environment

Install `python3-venv` if it isn't already available (e.g. `sudo apt install python3-venv`), then create a virtual environment to avoid modifying system packages:

</details>

```python
!python3 -m venv $HOME/.venv-rag-ollama
```

<!-- cell:15 -->

<a id="install-pytorch-optional"></a>

## 安装 PyTorch（可选） 

PyTorch 对于本教程是可选的。本节使用 PyTorch 实用程序进行验证。

<details markdown="1">
<summary>查看英文原文 · 单元 15</summary>

## Install PyTorch (optional) 

PyTorch is optional for this tutorial. This section uses PyTorch utilities for verification purposes.

</details>

```python
!. $HOME/.venv-rag-ollama/bin/activate && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

<!-- cell:17 -->

验证已安装的软件包列表：

<details markdown="1">
<summary>查看英文原文 · 单元 17</summary>

Verify the list of installed packages:

</details>

```python
!. $HOME/.venv-rag-ollama/bin/activate && pip list | grep torch
```

```text
pytorch-triton-rocm                      3.1.0
torch                                    2.5.1+rocm6.2
torchaudio                               2.5.1+rocm6.2
torchvision                              0.20.1+rocm6.2
```

<!-- cell:19 -->

验证 GPU 功能：

<details markdown="1">
<summary>查看英文原文 · 单元 19</summary>

Verify the GPU functionality:

</details>

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

<!-- cell:21 -->

<a id="install-llamaindex-and-dependencies"></a>

## 安装 LlamaIndex 和依赖项

使用以下命令安装 LlamaIndex 及相关软件包：

<details markdown="1">
<summary>查看英文原文 · 单元 21</summary>

## Install LlamaIndex and dependencies

Use the following command to install LlamaIndex and related packages:

</details>

```python
!. $HOME/.venv-rag-ollama/bin/activate && pip install llama-index llama-index-llms-ollama llama-index-embeddings-ollama llama-index-vector-stores-chroma chromadb
```

<!-- cell:23 -->

验证安装：

<details markdown="1">
<summary>查看英文原文 · 单元 23</summary>

Verify the installations:

</details>

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

<!-- cell:25 -->

<a id="build-the-rag-pipeline"></a>

## 构建 RAG 管道

本节介绍如何配置和构建 RAG 管道。

<a id="set-up-indexing-and-the-query-engine"></a>

### 设置索引和查询引擎

导入必要的库：

<details markdown="1">
<summary>查看英文原文 · 单元 25</summary>

## Build the RAG pipeline

This section explains how to configure and build the RAG pipeline.

### Set up indexing and the query engine

Import the necessary libraries:

</details>

```python
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
```

<!-- cell:27 -->

<a id="configure-embedding-and-llm-models"></a>

### 配置嵌入和LLM模型

LlamaIndex 实现 Ollama 客户端接口来与 Ollama 服务交互。在此示例中，它向 Ollama 请求嵌入和 LLM 服务。

<details markdown="1">
<summary>查看英文原文 · 单元 27</summary>

### Configure embedding and LLM models

LlamaIndex implements the Ollama client interface to interact with the Ollama service. In this example, it requests both embedding and LLM services from Ollama.

</details>

```python
# Set embedding model
emb_fn="nomic-embed-text"
Settings.embed_model = OllamaEmbedding(model_name=emb_fn)

# Set ollama model
Settings.llm = Ollama(model="llama3.1:8b", request_timeout=120.0)
```

<!-- cell:29 -->

<a id="download-data-for-rag"></a>

### 下载 RAG 数据

下载 PDF（例如 ROCm Radeon 文档）并将其保存到 `./data` 目录：

<details markdown="1">
<summary>查看英文原文 · 单元 29</summary>

### Download data for RAG

Download a PDF (for example, the ROCm Radeon documentation) and save it to the `./data` directory:

</details>

```python
!mkdir ./data && cd ./data && wget --recursive --level=1 --content-disposition --accept=pdf -np -nH --cut-dirs=6 https://rocm.docs.amd.com/_/downloads/radeon/en/latest/pdf/ && cd ..
```

<!-- cell:31 -->

SimpleDirectoryReader 是最常用的数据连接器。
为其提供输入目录或文件列表，它会根据文件扩展名选择最佳的文件阅读器。

<details markdown="1">
<summary>查看英文原文 · 单元 31</summary>

The SimpleDirectoryReader is the most commonly used data connector.
Provide it with an input directory or a list of files and it selects the best file reader based on the file extensions.

</details>

```python
documents = SimpleDirectoryReader(input_dir="./data/").load_data()

# Check the content
print(documents[10])
```

<!-- cell:33 -->

<a id="create-a-vector-dataset-with-chroma"></a>

### 使用 Chroma 创建矢量数据集

[Chroma DB](https://www.trychroma.com/) 是一个数据库，用于存储和查询 LLM 应用程序的嵌入、文档和元数据，与 LlamaIndex 良好集成。它通过获取 PDF 文件来创建矢量数据集。

<details markdown="1">
<summary>查看英文原文 · 单元 33</summary>

### Create a vector dataset with Chroma

[Chroma DB](https://www.trychroma.com/) is a database that stores and queries embeddings, documents, and metadata for LLM apps that integrates well with LlamaIndex. It creates the vector dataset by sourcing the PDF file.

</details>

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

<!-- cell:36 -->

<a id="create-the-query-engine"></a>

### 创建查询引擎

接下来，创建具有响应模式的查询引擎。根据您的具体需求选择响应模式。有关详细指导，请参阅 [LlamaIndex 响应模式文档](https://docs.llamaindex.ai/en/v0.10.19/module_guides/deploying/query_engine/response_modes.html)。

<details markdown="1">
<summary>查看英文原文 · 单元 36</summary>

### Create the query engine

Next, create the query engine with a response mode. Select the response mode based on your specific needs. For detailed guidance, see the [LlamaIndex response modes documentation](https://docs.llamaindex.ai/en/v0.10.19/module_guides/deploying/query_engine/response_modes.html).

</details>

```python
# Query your data
query_engine = vector_index.as_query_engine(response_mode="refine", similarity_top_k=10)
```

<!-- cell:38 -->

<a id="customize-the-query-prompts"></a>

## 自定义查询提示

定义特定于任务的提示：

<details markdown="1">
<summary>查看英文原文 · 单元 38</summary>

## Customize the query prompts

Define task-specific prompts:

</details>

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

<!-- cell:40 -->

<a id="query-examples"></a>

## 查询示例

运行以下查询：

<details markdown="1">
<summary>查看英文原文 · 单元 40</summary>

## Query examples

Run the following queries:

</details>

<!-- cell:41 -->

- 问题1：简述安装ROCm的步骤？

<details markdown="1">
<summary>查看英文原文 · 单元 41</summary>

- Query 1: Briefly describe the steps to install ROCm?

</details>

```python
response = query_engine.query("Briefly describe the steps to install ROCm?")
print(response)
```

<!-- cell:43 -->

- 问题2：哪一章是关于安装PyTorch的？

<details markdown="1">
<summary>查看英文原文 · 单元 43</summary>

- Query 2: Which chapter is about installing PyTorch?

</details>

```python
response = query_engine.query("Which chapter is about installing PyTorch?")
print(response)
```

<!-- cell:45 -->

- 问题 3：如何验证 PyTorch 安装？

<details markdown="1">
<summary>查看英文原文 · 单元 45</summary>

- Query 3: How to verify a PyTorch installation?

</details>

```python
response = query_engine.query("How to verify a PyTorch installation?")
print(response)
```

<!-- cell:47 -->

- 问题 4：ONNX 可以在 Radeon GPU 上运行吗？

<details markdown="1">
<summary>查看英文原文 · 单元 47</summary>

- Query 4: Could ONNX run on a Radeon GPU?

</details>

```python
response = query_engine.query("Could ONNX run on a Radeon GPU?")
print(response)
```

<!-- cell:49 -->

<a id="conclusion"></a>

## 结论

本教程演示如何在具有 ROCm 的 AMD Radeon GPU 上使用 LlamaIndex 和 Ollama 构建 RAG 管道。有关更多详细信息，请参阅不同组件的文档。

<details markdown="1">
<summary>查看英文原文 · 单元 49</summary>

## Conclusion

This tutorial demonstrates how to construct a RAG pipeline using LlamaIndex and Ollama on AMD Radeon GPUs with ROCm. For further details, see the documentation for the different components.

</details>
