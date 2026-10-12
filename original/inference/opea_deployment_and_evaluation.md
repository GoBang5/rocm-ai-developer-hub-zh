> [Upstream source](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/inference/opea_deployment_and_evaluation.ipynb) · Snapshot: `0071bfd4bd3bd9525405c09a6ae69c983b09312a`

# ChatQnA vLLM deployment and performance evaluation

**Author**: Yu Wang  
**Knowledge level**: Beginner

ChatQnA is a Retrieval-Augmented Generation (RAG) system that combines document retrieval with LLM inference. This tutorial provides a comprehensive guide for deploying ChatQnA using vLLM on AMD GPUs with ROCm support, as well as evaluating pipeline performance.

## Key features

Here are the benefits of using ChatQnA:

- **vLLM integration**: LLM serving with optimized inference on AMD Instinct™ GPUs
- **AMD GPU support**: ROCm-based GPU acceleration
- **Vector search**: Redis-based document retrieval
- **RAG pipeline**: Complete question-answering system
- **Performance monitoring**: Built-in metrics and evaluation tools

## Overview

This tutorial includes the following sections:

1. [Prerequisites](#prerequisites)
2. [Prepare the environment](#prepare-the-environment)
3. [System architecture](#system-architecture)
4. [Deployment guide](#deployment-guide)
5. [Performance evaluation](#performance-evaluation)
6. [Common issues and solutions](#common-issues-and-solutions)
7. [Advanced configuration](#advanced-configuration)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

This tutorial was developed and tested using the following setup. 

### Operating system

* **Ubuntu 22.04/24.04**: Ensure your system is running Ubuntu version 22.04 or 24.04.

### Hardware

* **AMD Instinct™ GPUs**: This tutorial was tested on an AMD Instinct MI300X GPU. Ensure you are using an AMD Instinct GPU or compatible hardware with ROCm support and that your system meets the [official requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

### Software

* **ROCm 6.3 or 6.4**: Install and verify ROCm by following the [ROCm install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html). 

### Hugging Face API access

* Obtain an API token from [Hugging Face](https://huggingface.co).

## Prepare the environment

This section creates a virtual environment and then starts the Jupyter server. 

### Set up a virtual environment

Start by creating a virtual environment:

``` bash
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

## System architecture

This section describes the ChatQnA architecture, including the services and data flow.

### Service components

The following diagram shows the complete ChatQnA system architecture.

**Architecture Overview:**
```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                               EXTERNAL ACCESS                                     │
│                                                                                   │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐   │
│   │   Web browser   │    │   API clients   │    │      Monitoring tools       │   │
│   │                 │    │                 │    │    (Grafana, Prometheus)    │   │
│   └─────────────────┘    └─────────────────┘    └─────────────────────────────┘   │
│           │                       │                           │                   │
│           │                       │                           │                   │
│           ▼                       ▼                           ▼                   │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐   │
│   │   Nginx Proxy   │    │   Backend API   │    │        Redis Insight        │   │
│   │   (Port 8081)   │    │   (Port 8890)   │    │         (Port 8002)         │   │
│   └─────────────────┘    └─────────────────┘    └─────────────────────────────┘   │
│           │                       │                           │                   │
│           │                       │                           │                   │
│           ▼                       ▼                           ▼                   │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐   │
│   │   Frontend UI   │    │     Backend     │    │   Redis Vector database     │   │
│   │   (Port 5174)   │    │     server      │    │         (Port 6380)         │   │
│   │   (React App)   │    │    (FastAPI)    │    │      (Vector storage)       │   │
│   └─────────────────┘    └─────────────────┘    └─────────────────────────────┘   │
│                                   │                           │                   │
│                                   │                           │                   │
│                                   ▼                           ▼                   │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │                             RAG PIPELINE                                    │  │
│  │                                                                             │  │
│  │  ┌───────────────────┐ ┌─────────────────────┐ ┌─────────────────────────┐  │  │
│  │  │ Retriever service │ │TEI embedding service│ │  TEI reranking service  │  │  │
│  │  │                   │ │                     │ │                         │  │  │
│  │  │   (Port 7001)     │ │    (Port 18091)     │ │      (Port 18809)       │  │  │
│  │  │                   │ │                     │ │                         │  │  │
│  │  │ • Vector search   │ │ • Text embedding    │ │ • Document reranking    │  │  │
│  │  │ • Similarity      │ │ • BGE model         │ │ • Relevance scoring     │  │  │
│  │  │   matching        │ │ • CPU inference     │ │ • CPU inference         │  │  │
│  │  └───────────────────┘ └─────────────────────┘ └─────────────────────────┘  │  │
│  │            │                      │                         │               │  │
│  │            │                      │                         │               │  │
│  │            ▼                      ▼                         ▼               │  │
│  │  ┌───────────────────────────────────────────────────────────────────────┐  │  │
│  │  │                           vLLM service                                │  │  │
│  │  │                           (Port 18009)                                │  │  │
│  │  │                                                                       │  │  │
│  │  │                  • High-performance LLM inference                     │  │  │
│  │  │                  • AMD GPU acceleration (ROCm)                        │  │  │
│  │  │                  • Qwen2.5-7B-Instruct model                          │  │  │
│  │  │                  • Optimized for throughput and latency               │  │  │
│  │  │                  • Tensor parallel support                            │  │  │
│  │  └───────────────────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                            │
│                                      │                                            │
│                                      ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │                            DATA PIPELINE                                    │  │
│  │                                                                             │  │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │  │
│  │  │   Dataprep      │    │   Model cache   │    │   Document storage      │  │  │
│  │  │   service       │    │   (./data)      │    │   (Redis Vector DB)     │  │  │
│  │  │   (Port 18104)  │    │                 │    │                         │  │  │
│  │  │                 │    │ • Downloaded    │    │ • Vector embeddings     │  │  │
│  │  │ • Document      │    │   models        │    │ • Metadata index        │  │  │
│  │  │   processing    │    │ • Model weights │    │ • Full-text search      │  │  │
│  │  │ • Text          │    │ • Cache storage │    │ • Similarity search     │  │  │
│  │  │   extraction    │    │ • Shared volume │    │ • Redis stack           │  │  │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────────┘
```

**Additional services:**
- **Dataprep service** (Port `18104`): Document processing and ingestion
- **Redis Insight** (Port `8002`): Database monitoring interface
- **Model cache** (`./data`): Shared volume for model storage

### Data flow

The pipeline for a new query follows these steps:

1. **User input**: A question is submitted using the frontend.
2. **Embedding**: The question is converted to a vector using the TEI service.
3. **Retrieval**: Similar documents are retrieved from the Redis vector database.
4. **Reranking**: The retrieved documents are reranked for relevance.
5. **LLM inference**: vLLM generates an answer using the retrieved context.
6. **Response**: The answer is returned to the user through the frontend.

## Deployment guide

To deploy ChatQnA, follow these steps:

### Step 1: Pull the source code from GitHub

First, clone the Open Platform for Enterprise AI (OPEA) GenAIExamples repository, which contains the ChatQnA implementation and other AI examples needed for your deployment.

```python
# Home directory
import os
HOME_DIR = os.getcwd()
# Open Platform for Enterprise AI (OPEA)
!git clone https://github.com/opea-project/GenAIExamples.git
```

Next, clone the LaunchPad repository that provides one-click deployment scripts and configuration files specifically designed for ChatQnA use cases on AMD GPU environments.

```python
# One click deployment scripts for the use case
!git clone https://github.com/Yu-amd/LaunchPad.git
```

Finelly, clone the GenAIEval evaluation repository, which contains the benchmarking tools you'll use to evaluate the ChatQnA system performance.

```python
# Pipeline performance evaluation harness
!git clone https://github.com/opea-project/GenAIEval.git
```

The LaunchPad project uses the same hierarchy as the OPEA project. Copy the LaunchPad scripts and YAML files from each directory to the corresponding directory in the OPEA folder.

```python
# Copy necessary scripts and configuration files to the OPEA directory
!cp {HOME_DIR}/LaunchPad/GenAIExamples/ChatQnA/docker_compose/amd/gpu/rocm/*.sh {HOME_DIR}/GenAIExamples/ChatQnA/docker_compose/amd/gpu/rocm/
!cp {HOME_DIR}/LaunchPad/GenAIExamples/ChatQnA/docker_compose/amd/gpu/rocm/*.yaml {HOME_DIR}/GenAIExamples/ChatQnA/docker_compose/amd/gpu/rocm/
!cp {HOME_DIR}/LaunchPad/GenAIExamples/ChatQnA/docker_compose/amd/gpu/rocm/.env {HOME_DIR}/GenAIExamples/ChatQnA/docker_compose/amd/gpu/rocm/
!cp -r {HOME_DIR}/LaunchPad/GenAIEval/evals/benchmarks/* {HOME_DIR}/GenAIEval/evals/benchmark/
```

### Step 2: Environment setup

Now navigate to the OPEA deployment directory where all the configuration files and scripts are located.

```python
# Navigate to the OPEA deployment directory
%cd {HOME_DIR}/GenAIExamples/ChatQnA/docker_compose/amd/gpu/rocm
```

You'll need to configure environment variable management. First, install the `python-dotenv` package which lets you load environment variables from an `.env` file. Then import the necessary modules and load the environment variables from the file that contains your configuration settings.

You also need to configure your Hugging Face API token, which is required to download the AI models used by the ChatQnA system. Replace `your_token_here` with the actual Hugging Face token.

```python
# Install and load .env
!pip install python-dotenv
# Configure Hugging Face token
!sed -i 's/CHATQNA_HUGGINGFACEHUB_API_TOKEN=""/CHATQNA_HUGGINGFACEHUB_API_TOKEN="YOUR_ACTUAL_TOKEN_HERE"/' .env
```

Set all the environment variables.

```python
# Load environment variables
from dotenv import load_dotenv
import os
load_dotenv()  # Loads variables from .env file
```

Now set up the vLLM environment using the provided script. This configures all the necessary components for high-performance LLM inference on AMD GPUs.

```python
# Setup vLLM environment
!./run_chatqna.sh setup-vllm
```

### Step 3: Deploy the workload

With the environment configured, you can now start the vLLM services. This launches all the necessary containers and services for the ChatQnA system.

```python
# Start vLLM services
import subprocess
subprocess.run(["./run_chatqna.sh", "start-vllm"], check=True)
```

Check the status of all the running services to ensure they started correctly and are functioning properly.

```python
# Check service status
!./run_chatqna.sh status
```

Monitor the vLLM service logs for 60 seconds to verify the service starts correctly. Review the logs for any initialization issues.

```python
# Check chatqna-vllm-service status
!timeout 200 docker logs -f chatqna-vllm-service
```

### Step 4: Verify the deployment

Verify that all Docker containers are running properly by checking their status and port mappings.

```python
# Check running containers
!docker ps
```

The next command sends a simple test message to the backend API to verify the ChatQnA service is working properly.

```python
# Test backend API
!curl -X POST http://localhost:8890/v1/chatqna \
  -H "Content-Type: application/json" \
  -d '{"messages": "Hello, how are you?"}'
```

### Step 5: Upload documents

Create a sample document and upload it to the system. This demonstrates how to feed documents into the ChatQnA system for retrieval and question answering.

```python
# Create a text file
!echo "Your document content here" > document.txt

# Upload the file
!curl -X POST http://localhost:18104/v1/dataprep/ingest \
  -H "Content-Type: multipart/form-data" \
  -F "files=@document.txt"
```

Now verify that the document was successfully uploaded and indexed by checking the contents of the Redis vector database.

```python
# Verify the upload worked
# Check if the document was indexed
!curl -X POST http://localhost:18104/v1/dataprep/get \
  -H "Content-Type: application/json" \
  -d '{"index_name": "rag-redis"}'
```

You can also upload multiple documents at once. Here's how to create and upload several documents simultaneously to build up your knowledge base.

```python
# For multiple documents
# Create multiple files
!echo "Document 1 content" > doc1.txt
!echo "Document 2 content" > doc2.txt

# Upload multiple files
!curl -X POST http://localhost:18104/v1/dataprep/ingest \
  -H "Content-Type: multipart/form-data" \
  -F "files=@doc1.txt" \
  -F "files=@doc2.txt"
```

## Performance evaluation

Performance evaluation helps you understand the following metrics:

- **Throughput**: Requests per second
- **Latency**: Response time
- **Accuracy**: Answer quality
- **Resource usage**: CPU, GPU, and memory utilization

### Step 1: Set up the evaluation environment

Navigate to the GenAIEval directory to set up and run your performance evaluation tests.

```python
# Navigate to evaluation directory
%cd {HOME_DIR}/GenAIEval
```

Install the required dependencies for the evaluation tools and set up the GenAIEval package in development mode.

```python
# Install evaluation dependencies
!pip install -r requirements.txt
!pip install -e .
```

### Step 2: Run the basic evaluation

Now navigate back to the ChatQnA deployment directory and run the performance evaluation tests on your deployed system.

```python
# Navigate back to GenAIExamples/ChatQnA/docker_compose/amd/gpu/rocm/
%cd {HOME_DIR}/GenAIExamples/ChatQnA/docker_compose/amd/gpu/rocm/
```

Now run the vLLM evaluation script, which tests the performance of your ChatQnA system, measuring metrics like throughput, latency, and response quality.

```python
# Run vLLM evaluation
!./run_chatqna.sh vllm-eval
```

### Step 3: Performance metrics

This section performs additional throughput and latency testing.

#### Throughput testing

Install Apache Bench (ab), which performs load testing and measures the throughput of the ChatQnA API under various conditions.

```python
# Install dependency
!apt install -y apache2-utils
```

Create a test file with a complex question to evaluate how well the system handles detailed, multi-part queries and generates comprehensive responses.

```python
# Create a complex test file
!echo '{"messages": "Can you provide a detailed explanation of how neural networks work, including the concepts of forward propagation, backpropagation, and gradient descent? Also explain how these concepts relate to deep learning and why they are important for modern AI systems."}' > test_data.json
```

Now use Apache Bench to run a load test simulating 100 concurrent requests with 10 simultaneous connections. This measures the system's throughput and performance under stress.

```python
# Test concurrent requests
!ab -n 100 -c 10 -p test_data.json -T application/json \
  http://localhost:8890/v1/chatqna
```

#### Latency testing

Create a detailed timing format file for `curl` to help measure various latency metrics including DNS lookup, connection time, and total response time for precise performance analysis.

```python
# Create curl-format.txt with the following content:
curl_format_content = """     time_namelookup:  %{time_namelookup}
        time_connect:  %{time_connect}
     time_appconnect:  %{time_appconnect}
    time_pretransfer:  %{time_pretransfer}
       time_redirect:  %{time_redirect}
  time_starttransfer:  %{time_starttransfer}
                     ----------
          time_total:  %{time_total}
          http_code:  %{http_code}
       size_download:  %{size_download}
      speed_download:  %{speed_download}"""

with open('curl-format.txt', 'w') as f:
    f.write(curl_format_content)

print("curl-format.txt has been created successfully!")
```

Now use `curl` with your detailed timing format to measure the precise response times for a single request. This provides granular insights into each step of the request processing pipeline.

```python
# Measure response times
!curl -w "@curl-format.txt" -X POST http://localhost:8890/v1/chatqna \
  -H "Content-Type: application/json" \
  -d '{"messages": "What is machine learning?"}'
```

### Step 4: Evaluation of results

The evaluation results include the following:

- **Response time**: Average, median, and 95th percentile
- **Throughput**: Requests per second
- **Accuracy**: Answer quality metrics
- **Resource usage**: CPU, GPU, and memory consumption

## Common issues and solutions

The performance results you collected could potentially indicate certain performance issues with the ChatQnA system.

### GPU memory errors

**Symptoms**: Out-of-memory or similar errors.

**Solution**: Reduce the batch size.

```python
# Reduce batch size in vLLM configuration
# Edit compose_vllm.yaml, modify vLLM service command:
--max-model-len 2048 --tensor-parallel-size 1
```

### Service startup failures

**Symptoms**: Services fail to start or remain in the `starting` state.

**Solution**: Check the logs.

```python
# Check logs for specific errors
!docker compose -f compose_vllm.yaml logs
```

Restart inactive services by passing `restart-vllm` to the `run_chatqna.sh` script.

```python
# Restart services
!./run_chatqna.sh restart-vllm
```

### Redis index issues

**Symptoms**: The retrieval service fails to find documents.

**Solution**: Fix the Redis index.

```python
!./fix_redis_index.sh
```

Then recreate the Redis index manually.

```python

!docker exec chatqna-redis-vector-db redis-cli FT.CREATE rag-redis ON HASH PREFIX 1 doc: SCHEMA content TEXT WEIGHT 1.0 distance NUMERIC
```

### Model download failures

**Symptoms**: Services fail to download models.

**Solution**: Verify the Hugging Face token.

```python
# Check HF token
!echo $CHATQNA_HUGGINGFACEHUB_API_TOKEN
```

Set your Hugging Face token manually.

```python
# Set token manually
!export CHATQNA_HUGGINGFACEHUB_API_TOKEN="your_token_here"
```

## Advanced configuration

This section covers advanced scenarios, such as how to use different models.

### Custom model configuration

Edit the `set_env_vllm.sh` file to use different models.

Run this command to change the LLM model:

```python
!export CHATQNA_LLM_MODEL_ID="Qwen/Qwen2.5-14B-Instruct"
```

Run this command to change the embedding model:

```python
!export CHATQNA_EMBEDDING_MODEL_ID="BAAI/bge-large-en-v1.5"
```

Run this command to change the reranking model:

```python
!export CHATQNA_RERANK_MODEL_ID="BAAI/bge-reranker-large"
```

## Troubleshooting

Use this section as a reference if the system isn't working as expected.

### Diagnostic commands

Run this command to check the system resources:

```python
!./detect_issues.sh
```

Run a quick test for the whole system using this command:

```python
!./quick_test_chatqna.sh eval-only
```

Run this command to check the service health:

```python
!docker compose -f compose_vllm.yaml ps
```

### Log analysis

View all logs:

```python
!docker compose -f compose_vllm.yaml logs
```

Review specific service logs:

```python
!docker compose -f compose_vllm.yaml logs -f chatqna-vllm-service
```

Check for errors:

```python
!docker compose -f compose_vllm.yaml logs | grep -i error
```

### GPU memory management

Determine GPU memory usage and clear memory if required.

#### Check GPU memory status

Check the current GPU memory usage.

**Note**: For ROCm 6.4 and earlier, use the `rocm-smi` command instead.

```python
# Check current GPU memory usage
# Expected output shows VRAM% and GPU% usage
# If VRAM% is high (>80%) but GPU% is low, memory may be fragmented
!amd-smi
```

#### Clear GPU memory (if necessary)

If you encounter GPU memory issues or high VRAM usage with low GPU utilization, try the commands in the following sections:

**Option 1: Kill GPU processes**

Find any processes that are using the GPU:

```python
!sudo fuser -v /dev/kfd
```

Kill the GPU-related processes:

```python

!sudo pkill -f "python|vllm|docker"
```

**Option 2: Restart GPU services**

Restart the `amdgpu` and related services:

```python
!sudo systemctl restart amdgpu
!sudo systemctl restart kfd
```

**Option 3: System reboot**

If the other methods don't work, reboot the system. This is the most reliable way of dealing with GPU memory issues.

```python
# Note: If you're on a remote server, wait approximately 30 seconds to 1 minute
# before attempting to SSH back into the server
!sudo reboot
```

After clearing GPU memory, verify memory is available again.

**Note**: For ROCm 6.4 and earlier, use the `rocm-smi` command instead.

```python

# Check GPU memory is now available
# Expected: VRAM% should be low (<20%) and GPU% should be 0%
!amd-smi
```

## Conclusion

This tutorial provides a comprehensive guide for deploying ChatQnA with vLLM on AMD GPUs and performing detailed performance evaluation. The ChatQnA system offers:

- **High performance**: vLLM-optimized inference
- **Scalability**: Docker-based microservices architecture
- **Monitoring**: Built-in performance metrics
- **Flexibility**: Configurable models and parameters

For additional support or advanced configurations, see the [project documentation](https://github.com/opea-project/GenAIExamples/tree/main/ChatQnA) or create an issue in the repository.

### Next steps

1. **Customize models**: Experiment with different LLM and embedding models.
2. **Scale deployment**: Add multiple GPU nodes for higher throughput.
3. **Optimize performance**: Fine-tune vLLM parameters for your specific use case.
4. **Monitor production**: Set up comprehensive monitoring for production deployments.

**Note**: This tutorial assumes you have the necessary permissions and that all required software is installed. For production deployments, consider additional security measures and monitoring solutions.
