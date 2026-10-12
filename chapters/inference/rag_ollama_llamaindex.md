> 原文作者：Alex He  
> [AMD 官方教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/rag_ollama_llamaindex.html) · [固定版本源码](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/inference/rag_ollama_llamaindex.ipynb)  
> 上游版本：`0071bfd4bd3bd9525405c09a6ae69c983b09312a` · 同步日期：2026-10-12  
> 翻译状态：机器初译，待逐篇人工校对；代码与命令保持原文，未在 AMD GPU 上运行验证。

[中文 Notebook](../../notebooks/inference/rag_ollama_llamaindex.ipynb) · [英文原文](../../original/inference/rag_ollama_llamaindex.md) · [中英对照](../../bilingual/inference/rag_ollama_llamaindex.md)

---

<a id="constructing-a-rag-system-using-llamaindex-and-ollama"></a>

# 使用 LlamaIndex 和 Ollama 构建 RAG
**作者**：Alex He
**知识水平**：初学者

AMD Radeon™ GPU 得到 [ROCm](https://rocm.docs.amd.com/en/latest/index.html) 的正式支持，确保与行业标准软件框架的兼容性。此 Jupyter Notebook利用由 ROCm 提供支持的 [Ollama](https://ollama.com/) 和 [LlamaIndex](https://docs.llamaindex.ai/en/stable/) 来构建 检索增强生成 (RAG) 应用程序。 LlamaIndex 有助于创建从读取 PDF 到索引数据集和构建查询引擎的管道，而 Ollama 为 大语言模型 (LLM) 推理 提供后端服务。

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

```python
!curl -fsSL https://ollama.com/install.sh | sh
```

启动 Ollama 并验证其正在运行：

**注意**：Ollama 安装指南可在[此处](https://github.com/ollama/ollama/blob/main/docs/linux.mdx) 获取。

```python
!sudo systemctl start ollama
!sudo systemctl status ollama --no-pager
```

<a id="download-the-models"></a>

## 下载模型

使用 Ollama 拉取 RAG 所需的模型：

**重要**：如果 Ollama 服务器作为上一步中的前台进程运行，则必须在新实例中运行此Notebook的其余部分。

```python
!ollama pull nomic-embed-text
!ollama pull llama3.1:8b
```

验证下载的模型：

```python
!ollama list llama3.1
```

```text
NAME           ID              SIZE      MODIFIED     
llama3.1:8b    42182419e950    4.7 GB    2 months ago    
```

有关更多详细信息，请参阅 [Ollama 文档](https://github.com/ollama/ollama)。

**注意**：[此处](https://ollama.com/search) 可以使用替代型号。

<a id="set-up-a-virtual-environment"></a>

## 设置虚拟环境

如果 `python3-venv` 尚不可用，请安装它（例如 `sudo apt install python3-venv`），然后创建一个虚拟环境以避免修改系统包：

```python
!python3 -m venv $HOME/.venv-rag-ollama
```

<a id="install-pytorch-optional"></a>

## 安装 PyTorch（可选） 

PyTorch 对于本教程是可选的。本节使用 PyTorch 实用程序进行验证。

```python
!. $HOME/.venv-rag-ollama/bin/activate && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

验证已安装的软件包列表：

```python
!. $HOME/.venv-rag-ollama/bin/activate && pip list | grep torch
```

```text
pytorch-triton-rocm                      3.1.0
torch                                    2.5.1+rocm6.2
torchaudio                               2.5.1+rocm6.2
torchvision                              0.20.1+rocm6.2
```

验证 GPU 功能：

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

<a id="install-llamaindex-and-dependencies"></a>

## 安装 LlamaIndex 和依赖项

使用以下命令安装 LlamaIndex 及相关软件包：

```python
!. $HOME/.venv-rag-ollama/bin/activate && pip install llama-index llama-index-llms-ollama llama-index-embeddings-ollama llama-index-vector-stores-chroma chromadb
```

验证安装：

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

<a id="build-the-rag-pipeline"></a>

## 构建 RAG 管道

本节介绍如何配置和构建 RAG 管道。

<a id="set-up-indexing-and-the-query-engine"></a>

### 设置索引和查询引擎

导入必要的库：

```python
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
```

<a id="configure-embedding-and-llm-models"></a>

### 配置嵌入和LLM模型

LlamaIndex 实现 Ollama 客户端接口来与 Ollama 服务交互。在此示例中，它向 Ollama 请求嵌入和 LLM 服务。

```python
# Set embedding model
emb_fn="nomic-embed-text"
Settings.embed_model = OllamaEmbedding(model_name=emb_fn)

# Set ollama model
Settings.llm = Ollama(model="llama3.1:8b", request_timeout=120.0)
```

<a id="download-data-for-rag"></a>

### 下载 RAG 数据

下载 PDF（例如 ROCm Radeon 文档）并将其保存到 `./data` 目录：

```python
!mkdir ./data && cd ./data && wget --recursive --level=1 --content-disposition --accept=pdf -np -nH --cut-dirs=6 https://rocm.docs.amd.com/_/downloads/radeon/en/latest/pdf/ && cd ..
```

SimpleDirectoryReader 是最常用的数据连接器。
为其提供输入目录或文件列表，它会根据文件扩展名选择最佳的文件阅读器。

```python
documents = SimpleDirectoryReader(input_dir="./data/").load_data()

# Check the content
print(documents[10])
```

<a id="create-a-vector-dataset-with-chroma"></a>

### 使用 Chroma 创建矢量数据集

[Chroma DB](https://www.trychroma.com/) 是一个数据库，用于存储和查询 LLM 应用程序的嵌入、文档和元数据，与 LlamaIndex 良好集成。它通过获取 PDF 文件来创建矢量数据集。

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

<a id="create-the-query-engine"></a>

### 创建查询引擎

接下来，创建具有响应模式的查询引擎。根据您的具体需求选择响应模式。有关详细指导，请参阅 [LlamaIndex 响应模式文档](https://docs.llamaindex.ai/en/v0.10.19/module_guides/deploying/query_engine/response_modes.html)。

```python
# Query your data
query_engine = vector_index.as_query_engine(response_mode="refine", similarity_top_k=10)
```

<a id="customize-the-query-prompts"></a>

## 自定义查询提示

定义特定于任务的提示：

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

<a id="query-examples"></a>

## 查询示例

运行以下查询：

- 问题1：简述安装ROCm的步骤？

```python
response = query_engine.query("Briefly describe the steps to install ROCm?")
print(response)
```

- 问题2：哪一章是关于安装PyTorch的？

```python
response = query_engine.query("Which chapter is about installing PyTorch?")
print(response)
```

- 问题 3：如何验证 PyTorch 安装？

```python
response = query_engine.query("How to verify a PyTorch installation?")
print(response)
```

- 问题 4：ONNX 可以在 Radeon GPU 上运行吗？

```python
response = query_engine.query("Could ONNX run on a Radeon GPU?")
print(response)
```

<a id="conclusion"></a>

## 结论

本教程演示如何在具有 ROCm 的 AMD Radeon GPU 上使用 LlamaIndex 和 Ollama 构建 RAG 管道。有关更多详细信息，请参阅不同组件的文档。
