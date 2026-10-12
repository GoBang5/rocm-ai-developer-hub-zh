> [Upstream source](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/inference/agentic_ai_profiling.ipynb) · Snapshot: `0071bfd4bd3bd9525405c09a6ae69c983b09312a`

# Profiling and optimizing an AI agent on AMD Instinct GPUs

**Author**: Shailen Sobhee, Jereshea John Mary, Sabira Shaik  
**Knowledge level**: Intermediate  
**Publication date:** September 28, 2026 

In this hands-on tutorial, you will learn to profile and optimize an AI agent. You will run
a real AI agent, watch where it spends its time, identify and optimize a slow step, 
and prove the speed-up with hardware telemetry. No prior profiling experience is assumed.

## Background

This section introduces the key concepts and tools used in this tutorial.

### Agentic AI

Unlike a plain chatbot, an **agent** can **plan and complete multi-step tasks on its
own**. The agent also decides which **tools** to use and in what order to call them to achieve its goal.

> **For those new to agentic AI,** here are two terms to know:
>
> **Tool**: A single action the agent can perform, exposed as a callable function (e.g., "convert this text to speech" or "read a file"). At each step the agent picks the tool that fits.   
>
> **Skill:** A higher-level, reusable ability. It is a packaged set of instructions that might combine several tools to handle a larger task.

[**Hermes Agent**](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart)
is an open-source, autonomous AI agent framework by [**Nous Research**](https://nousresearch.com/). It ships with built-in tools, and you can add to it **custom tools** like the [**Kokoro TTS**](https://kokorottsai.com/) tool used in this tutorial. 

### Why profiling matters

When an agent runs, some steps are fast and some are slow. Profiling measures each step so you can see exactly where the agent spends its time.

An agent is only as fast as the tools it calls. A single slow tool can slow down an entire run, even when the AI model driving it is fast. Finding and fixing that slow tool often makes the difference between a sluggish agent and a responsive one.

## What you will learn

In this tutorial, you will learn to profile a text-to-speech (TTS) workflow, which is illustrated in the following diagram:

<p align="center">
<img alt="Pipeline: an input text file flows into the Hermes Agent, which calls a TTS tool, which produces audio output. The workshop focus is profiling and optimization." src="../assets/embedded-28d8fe200828d4fce6c0.png" width="92%">
</p>
By the end of this session you will be able to:

- Run a Hermes agent and capture its telemetry
- Read the telemetry dashboard to spot a slow tool
- Swap in an optimized version of that tool
- Compare before and after, and see how hardware usage has changed

## Prerequisites

This tutorial was developed and tested with the setup below.

### Operating system
**Ubuntu 24.04:** Ensure this is the Linux distribution your system is running on.

### Hardware
**An AMD Instinct&trade; GPU:** This tutorial was tested on a single MI300X GPU (192 GB VRAM), which comfortably hosts both the Muse-Glimmer-30B model and the Kokoro TTS model all at once. Ensure you are using an AMD Instinct GPU or compatible hardware with ROCm support and that your system meets the [official requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

### Software

**ROCm 7.14.1 or later**: Install and verify ROCm by following the [ROCm install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html). After installation, confirm your setup using:

```bash
amd-smi
```

This command lists your AMD GPUs with relevant details.

**vLLM ROCm image:** The agent's model is served with the vLLM inference and serving engine using the `vllm/vllm-openai-rocm` image, which includes Muse-Glimmer-30B support. This is not an AMD-released image, but an upstream image built by the open-source community and maintained with the help of AMD engineers.

**Python 3.12:** It runs the Kokoro server, MLflow, and this notebook.

## Environment setup

The whole tutorial runs inside one container started from the official upstream `vllm/vllm-openai-rocm` image. That image already provides vLLM, PyTorch, AITER, and the ROCm libraries. Everything else the tutorial needs is installed inside this notebook by running the cells below, so there is no custom image to pull or build.

### 1. Launch the container and open JupyterLab

Run this single command from a host shell. It starts the official image, installs JupyterLab inside it, and serves this notebook:

```bash
docker run -d --name amd-agentic-ai-profiling \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined --group-add video \
  --ipc=host --shm-size 16G --network=host \
  -v "$HOME/hf_cache:/root/.cache/huggingface" \
  --entrypoint bash \
  vllm/vllm-openai-rocm:latest -c \
  'python3 -m pip install --break-system-packages -q jupyterlab==4.5.7 && \
   jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
     --ServerApp.root_dir=/root'
```
> **Why mount `$HOME/hf_cache`?** Docker creates the host side of a bind mount owned by the current user if the directory does not exist yet. Mounting a fresh `$HOME/hf_cache` means the Hugging Face cache is user owned from the start, so model downloads work even on clusters where you cannot `sudo chown` a root-created `~/.cache/huggingface`. The container sees it as `/root/.cache/huggingface`.

Then read the JupyterLab URL (with its access token) from the container logs with this command:

```bash
docker logs amd-agentic-ai-profiling 2>&1 | grep -m1 'http://127.0.0.1:8888'
```

Open that link in your browser, then open this notebook (`agentic_ai_profiling.ipynb`) using the JupyterLab file browser and run the rest of the tutorials from there.

> **If the GPU is on a remote machine** you accessed via SSH, open a second terminal on your local system and 
> forward the port so you can reach the remote machine and its browser UI from your local browser:
>
> ```bash
> ssh -L 8888:localhost:8888 <user>@<gpu-host>
> ```
> Open the JupyterLab URL (with its access token) in your browser, then use JupyterLab's **Upload Files** button to
> upload this notebook (`agentic_ai_profiling.ipynb`) to the remote environment. Open the uploaded notebook and run
> the rest of the tutorial from there.

This notebook is available at the [AI Developer Hub GitHub repository](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/inference/agentic_ai_profiling.ipynb).

### 2. Fetch the tutorial's support files

The tutorial uses a few support files: the backend setup script (`utils/setup_backend.sh`), the profiling **dashboard** (`utils/hermes_profiler.py`), the local **Kokoro TTS server** (`utils/kokoro_server.py`) and its start/clear scripts, the dashboard theme, and the custom **`kokoro_tts` agent tool** (`custom_tools/kokoro_tts_tool.py`). The cell below fetches them from a community public repository into the folders `utils/` and `custom_tools/` next to this notebook.

Run the following cell once. It prints each file it fetches, and it is safe to re-run.

```python
%%bash
set -euo pipefail

REPO_URL="https://github.com/shailensobhee/amd-agentic-ai-profiling-workshop.git"
BRANCH="single-container"
CLONE_DIR=".nb_support_checkout"

# Fetch only the support files the later cells import or deploy, not the whole repo.
rm -rf "$CLONE_DIR"
git clone --quiet --depth 1 --branch "$BRANCH" --filter=blob:none --sparse "$REPO_URL" "$CLONE_DIR"
( cd "$CLONE_DIR" && git sparse-checkout set utils custom_tools >/dev/null )

# Place the files the notebook uses. setup_backend.sh is the backend plumbing
# (downloads and launches Prometheus, Grafana, MLflow, vLLM, and the Hermes
# agent); its tutorial-facing configuration is set in the next cell.
mkdir -p utils/.streamlit custom_tools
cp "$CLONE_DIR/utils/setup_backend.sh"         utils/
cp "$CLONE_DIR/utils/hermes_profiler.py"        utils/
cp "$CLONE_DIR/utils/kokoro_server.py"          utils/
cp "$CLONE_DIR/utils/start_kokoro_server.sh"    utils/
cp "$CLONE_DIR/utils/clear_cache.sh"            utils/
cp "$CLONE_DIR/utils/.streamlit/config.toml"    utils/.streamlit/
cp "$CLONE_DIR/custom_tools/kokoro_tts_tool.py" custom_tools/
rm -rf "$CLONE_DIR"

echo "[OK] Support files ready:"
echo "     utils/setup_backend.sh           (backend setup: services + telemetry)"
echo "     utils/hermes_profiler.py         (profiling dashboard)"
echo "     utils/kokoro_server.py           (local Kokoro TTS server)"
echo "     utils/start_kokoro_server.sh     (starts the Kokoro server)"
echo "     utils/clear_cache.sh             (clears the GPU kernel cache)"
echo "     utils/.streamlit/config.toml     (dashboard theme)"
echo "     custom_tools/kokoro_tts_tool.py  (custom kokoro_tts agent tool)"
```

### 3. Configure the backend

The backend is set up by `utils/setup_backend.sh`, which you downloaded in the previous cell. The setup logic is intentionally kept out of the notebook to avoid a long wall of shell code. Instead, the cell below exposes the settings you are most likely to customize: the model used by the agent, the service ports, the GPU memory fraction, and the Python packages missing from the official image. The script reads these values from environment variables, making this cell the single place to adjust the configuration. Everything else, including downloading and launching Prometheus, Grafana, MLflow, vLLM, and the Hermes agent, is implementation detail that remains in the script.

To inspect the full script at any time, open `utils/setup_backend.sh` in the JupyterLab file browser or run `!cat utils/setup_backend.sh` in a new cell.

```python
# ---------------------------------------------------------------------------
# Configure the backend
# ---------------------------------------------------------------------------
# These are the knobs the tutorial cares about. utils/setup_backend.sh reads
# them from the environment, so this is the one place you change the model, the
# service ports, or the GPU memory fraction.
import os

# --- Service configuration (the setup script reads these) ------------------
os.environ["HERMES_MODEL"]           = "meta-models/Muse-Glimmer-30B"  # agent model, served by vLLM
os.environ["VLLM_PORT"]              = "8001"   # vLLM OpenAI-compatible API
os.environ["MLFLOW_PORT"]            = "5004"   # MLflow tracking server (traces)
os.environ["PROM_PORT"]              = "9090"   # Prometheus (metrics store, OTLP receiver)
os.environ["GRAFANA_PORT"]           = "3000"   # Grafana (metrics UI)
os.environ["GPU_MEMORY_UTILIZATION"] = "0.80"   # fraction of VRAM vLLM may use

# --- Python packages the official image does not ship ----------------------
# OpenTelemetry must stay a coherent 1.40.0 set: the image ships otel 1.40.0 and
# lmcache caps opentelemetry-api at <=1.40.0, so bumping only part of the stack
# breaks 'pip check'. The setup script reuses OTEL_PINS for its plugin install.
os.environ["OTEL_PINS"] = (
    "opentelemetry-api==1.40.0 opentelemetry-sdk==1.40.0 opentelemetry-proto==1.40.0 "
    "opentelemetry-semantic-conventions==0.61b0 "
    "opentelemetry-exporter-otlp-proto-common==1.40.0 "
    "opentelemetry-exporter-otlp-proto-grpc==1.40.0 "
    "opentelemetry-exporter-otlp-proto-http==1.40.0 "
    "opentelemetry-exporter-otlp==1.40.0"
)

os.environ["PIP_BREAK_SYSTEM_PACKAGES"] = "1"
os.environ["PIP_ROOT_USER_ACTION"]      = "ignore"

# Install the tutorial's Python packages. setup_backend.sh installs these too
# when they are missing (so it still runs standalone), but doing it here keeps
# the dependency list visible.
get_ipython().system(
    'python3 -m pip install -q '
    'kokoro soundfile fastapi "uvicorn[standard]" '
    '"streamlit>=1.30" "mlflow>=3.0.0" plotly pandas ipywidgets matplotlib kaleido '
    '$OTEL_PINS '
    'psutil requests'
)
print("[OK] Backend configuration set and Python packages installed.")
```

### 4. Start the tutorial backend

The cell below runs `utils/setup_backend.sh` with the configuration set in the preview cell. It starts the services the tutorial profiles: the vLLM-served agent model, MLflow for the execution traces, and Prometheus and Grafana for the CPU and GPU metrics. The first run downloads the agent model (about 60 GB), so it might take several minutes to complete. The cell prints `Backend ready` when everything is up. It is safe to re-run the cell.

```python
# Start the tutorial backend inside this container.
#
# The configuration and Python packages were set in the cell above. This cell
# runs utils/setup_backend.sh (fetched from the repository earlier), which
# downloads and starts the services the tutorial profiles:
#   * vLLM (the agent's model)    : the model that plans and picks tools
#   * MLflow                      : stores the agent's execution traces
#   * Prometheus (upstream)       : stores CPU/GPU metrics; ingests them over
#                                   OTLP directly (no separate collector)
#   * Grafana (upstream)          : the metrics UI
#   * Hermes + hermes-otel plugin : the agent and its telemetry
#   * Streamlit dashboard         : the profiling view over traces and metrics
#
# It is safe to re-run; each service is restarted cleanly.
get_ipython().system('bash utils/setup_backend.sh')
```

Here are the details on each of the services the setup starts:

| Service | Role |
| :--- | :--- |
| **Hermes backend** (vLLM &middot; Muse-Glimmer-30B) | The agent's "brain": the model that plans and picks tools. |
| **Hermes OTel** | The plugin that instruments the agent. The setup writes its config file with two OpenTelemetry backends: it sends execution **traces** (spans, timings, tokens) to the MLflow tracking server, and hardware **metrics** to Prometheus. The setup configures it to sample `psutil` (CPU) and `amdsmi` (GPU) every 100 ms (much finer than the plugin's default), so the timelines have the resolution to see per-tool activity. |
| **MLflow tracking server** | Stores the execution **traces** the dashboard visualizes. |
| **Prometheus + Grafana** | Receives the CPU/GPU **metrics** over OTLP and stores them in Prometheus, which the dashboard queries for the utilization timelines: system-wide GPU%, the Hermes process (plus children) CPU%, and per-tool CPU/GPU%. |
| **Telemetry dashboard** | A custom Streamlit page that reads traces from MLflow and metrics from Prometheus to give one clear view of each run. |

> **About the LLM model: Muse-Glimmer-30B** is a dense vision-language model built for agentic work, trained on more than 100 
> languages. It features a 52-layer text decoder (hidden size 6656) and a ~1.8B-parameter ViT-G/14 perception encoder, with a 
> trained context length of 128K tokens. The model uses BF16 precision and is licensed under the Apache 2.0 
> open-source license. Its knowledge cutoff is January 4, 2026.

### 5. Check your setup

Confirm the `hermes` command is available in 
this notebook's environment. The cell below adds the usual install locations to
`PATH`, prints the path to the Hermes binary, then prints `Hermes is ready.`

**If nothing prints,** it means the notebook cannot find Hermes. Make sure the backend setup cell above has finished and Hermes has been installed in this container.

```python
import os

# Hermes may live in either location depending on how it was installed:
#   /usr/local/bin  -> install performed by the backend setup cell
#   ~/.local/bin    -> per-user pip install
# A JupyterLab kernel does not always inherit the login shell's PATH, so add
# both and let `which` report the one actually in use.
for _p in ("/usr/local/bin", os.path.expanduser("~/.local/bin")):
    if _p not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] += os.pathsep + _p

!which hermes && echo "Hermes is ready."
```

### 6. Prepare your input text

TTS is the example use case to be profiled, so a passage for speech 
synthesis is needed. Run the cell below to let Hermes itself write the input passage and save
it to `input_text.txt`, which is then passed to the TTS tool.

```python
!hermes chat --yolo --oneshot -q "Write ONE single continuous paragraph of about 1,000 words (roughly 8000 characters) on the topic of AMD GPUs. It must be a single block of flowing prose: do NOT number the sentences, do NOT put each sentence on its own line, and do NOT use any line breaks, headings, bullet points, lists, quotes, code, or special symbols. Use normal sentence punctuation (periods and commas) so it reads naturally for text-to-speech. Write the whole passage as one block with no newline characters. Save it to 'input_text.txt' in the current directory, overwriting existing content, using your write/file tool. Do not read any other file."
```

When the text preparation is finished, the cell prints a short summary of the generated paragraph, with information on the Hermes session that created it. 

> **Make it your own:** Feel free to create your own passage and
> to vary its length and save it as `input_text.txt`. Longer input makes the performance gain by batching 
> (implemented later in the notebook) more apparent.

## Step 1: Establish a baseline with Edge TTS

The profiling exercise starts with running **Edge TTS**, the
text-to-speech provider that Hermes uses by default.

The command below asks the agent to read the generated `input_text.txt` and speak
it.

```python
!hermes chat --yolo --oneshot -q "Convert the entire text in input_text.txt to audio and save it in output_audio.mp3"
```

After Hermes finishes the audio generation, the cell prints, among other things, a summary of the run, and it saves the output as `output_audio.mp3` (or to multiple files if the audio clip is long).

## Step 2: Profiling

In the output of the last cell, you will notice log entries similar to these:

```text
[hermes-otel] ✓ mlflow at http://127.0.0.1:5004/v1/traces (traces only)
[hermes-otel] ✓ lgtm at http://127.0.0.1:9090/api/v1/otlp/v1/traces (metrics only)
[hermes-otel] ✓ Live dashboard store active
[hermes-otel] ✓ Host metrics sampler on (every 100 ms, gpu=amd)
[hermes-otel] Registered 13 hooks
```

These appear because `utils/setup_backend.sh` configures the `hermes-otel` plugin to send this
session's execution traces to the MLflow server (its CPU/GPU metrics go to
Prometheus separately). The telemetry dashboard in the next section reads
both: traces from MLflow and metrics from Prometheus.

### Launching the profiling dashboard

To make the MLflow data easier to read, there is a custom Streamlit dashboard on
port `8501`, started for you by `utils/setup_backend.sh`. The cells below show you the **Overview** tab of the dashboard, and 
resolve your server address to give you a direct link to the full dashboard.

```python
# NOTE: Skip this cell if you are using the default AMD hosted-notebook proxy.
# If you are running locally or using a custom proxy, uncomment the next line to set the appropriate base below:
# os.environ["HERMES_PROXY_BASE"] = ""  # Use "" for local 127.0.0.1 links, or "https://your-custom-proxy"
```

```python
# This cell shows an overview tab alone; for a detailed view, click on the links provided when running this cell.
import importlib, sys, os, logging
sys.path.insert(0, os.path.abspath("utils"))
logging.disable(logging.WARNING)          # mute Streamlit's import-time warnings
import hermes_profiler
importlib.reload(hermes_profiler)          # pick up edits without a kernel restart
logging.disable(logging.NOTSET)

SESSION_ID = None   # a run ID, or None to auto-fetch the latest session
hermes_profiler.show_session_overview(SESSION_ID)
```

> **About the links this cell prints.** On the default AMD hosted-notebook proxy
> the cell prints the two service links directly (the dashboard and the raw MLflow
> view), and those are the ones to use. If you are running the tutorial backend locally or forwarding
> the port (`ssh -L 8501:localhost:8501`) by setting
> `os.environ["HERMES_PROXY_BASE"] = ""`, the cell additionally shows a
> `localhost` link, which you should use. If you are running the tutorial backend on a remote
> machine that you access directly and port `8501` is reachable from your network, use the
> server-IP link. If a link does not load, see the **[Troubleshooting](#troubleshooting)** section
> at the end of this notebook.

**The dashboard has five tabs:**

1. **Overview:** Plots the span waterfall for the agent's flow alongside CPU and
   GPU utilization at each span. A toggle switches between the standalone CPU/GPU
   timeline (the default) and the full-session waterfall correlated with it. A
   tool-breakdown table sits below the chart.
2. **CPU / GPU separate:** Shows the CPU and GPU graphs individually, with an
   option to view the raw `.csv` files used to generate the **Overview** charts.
3. **Context & tools:** Displays context growth throughout the session, a per-turn breakdown, and the outcome of each tool invocation, including failed calls.
4. **Traces:** Provides a direct MLflow link for each turn in the session.
5. **Analysis:** Feeds the MLflow traces plus each tool's execution time to the
   local `hermes` CLI and reports how the agent could be improved. Depending on
   the length of the traces, this can take around five minutes.

> Besides the link to the dashboard, the cells above also generate a link to the MLflow traces.
> The following steps open the raw MLflow interface, where you can browse detailed traces and logs (skip them if 
> you only want the high-level overview the dashboard gives you):
> 
> 1. Open the MLflow interface with the provided link.
> 2. Open the **Traces** tab in the left panel and find your recent request.
> 3. In parallel, open the **Evaluation Runs** tab and find your session ID there. The session ID
>   is printed in the cell output after each Hermes run, next to `[hermes-otel]`.

## Step 3: Analyzing the run

After each Hermes execution, review the telemetry to understand where time was
spent and which tool drove the latency:

- Every Hermes run is recorded as a session with a unique session ID: its
  execution traces are logged to MLflow and its CPU/GPU metrics to `otel-lgtm`.
- In the left pane of the dashboard, click **Fetch** to load IDs of the recorded runs. The most recent
  run appears at the top, followed by older ones.
- Select the run you want (usually the latest) under **Session ID**, then click **Load / Reload**.
  The dashboard draws a picture of what happened.

> The loop is always the same: run the agent, click **Fetch**, select the
> latest run, click **Load / Reload**, and inspect.

Once the run loads, explore what the dashboard shows for this execution:

- The timeline of the run
- When the TTS tool ran, and for how long
- Hardware metrics over time (GPU and CPU utilization)
- Overall execution latency

The following is a screen capture of the **Overview** tab of the telemetry dashboard for a sample run:

<div align="center">

![The AMD Agent Telemetry dashboard Overview tab for a Hermes session. A span waterfall shows the agent, LLM and API spans plus the text_to_speech tool span highlighted in orange as the longest at 24.44 seconds, and a CPU/GPU utilization time series below tracks hardware use across the run.](../assets/embedded-f62461df875ec9f86138.png)

</div>

The orange `text_to_speech` span is the longest single step, and the utilization chart shows the GPU is mostly idle while it runs. That gap is the bottleneck you will fix in Step 5.

> **How this works under the hood**
>
> The dashboard builds this view from two sources: the session's spans come from the
> MLflow traces, and the CPU/GPU timeline is queried from the metrics stored in
> `otel-lgtm` (Prometheus). It compiles both into one human-readable summary.

### Edge TTS observations

In the telemetry dashboard, find the Hermes session id for the Edge TTS run you
just did, then look at the audio Edge produced and note a few things:

- **No local setup.** Edge is cloud-based. Hermes sends text to the service and
  receives audio back, which makes it a convenient baseline: nothing to install,
  nothing to configure.
- **Input-length limit.** Edge TTS has a 5,000-character input limit. The
  Hermes wrapper handles longer inputs by splitting them into chunks (a
  10K-character input becomes two chunks, and so on, depending on the chunking
  logic).
- **Audio stitching.** The chunks are normally combined into one output, but for
  very large inputs they might not stitch successfully, resulting in several audio files
  for a single request.
- **Short inputs.** For shorter inputs, Edge TTS works well and is convenient.
- **Privacy.** Edge is cloud-based. In most real-world deployments, the texts are sent from a 
local machine to the cloud for processing. For privacy-sensitive workloads, a local model is preferable.

> **Why this step can produce messy, repeated tool calls.** The built-in
> `text_to_speech` tool accepts text only *inline* (there is no file-path
> parameter). For a long passage the agent cannot pass the whole text in one call,
> so it often improvises with extra steps (chunked file reads, `wc`/`head`/`cat`,
> temporary copies, small Python snippets, etc.) while working around the limit.

The length limit, the stitching edge cases, the privacy consideration, and the
network round-trip on every request are together reasons to move to a local
model next.

## Step 4: Local TTS with Kokoro

**Kokoro** is a text-to-speech model that runs entirely on the local machine on a local GPU. Because inference happens locally, your text never leaves the system, making it a natural fit
for privacy-sensitive workloads. It also means performance will depend on how well the
tool uses the local hardware, which is the bottleneck to profile.

While this whole tutorial runs on a single machine inside a single container, this
step illustrates how you install Kokoro as your local TTS and how that changes the
profiling and optimization strategy for the workflow. Running everything in one place
(the model server, the local TTS, and the telemetry stack) is a deliberate teaching
setup: it makes the local-versus-cloud bottleneck observable end to end on one GPU. In
a production deployment you would typically run the TTS as its own service rather than
registering it as a Hermes backend tool, but the profiling loop you learn here (measure,
find the bottleneck, optimize, measure again) is exactly the same.

### Kokoro ships as a Python package, not as a service

Kokoro is distributed as a `pip` package. You install it, import it, and call it
inside your own process:

```python
from kokoro import KPipeline
pipeline = KPipeline(lang_code="a")      # loads the model
audio = pipeline(text)                   # synthesizes
```

That is a library. It works well for a one-off script that synthesizes some text
and exits, but it is not a good fit for this tutorial because the model will reload each time the process starts. 
The `KPipeline(...)` call loads Kokoro-82M into GPU memory, so each new process pays the model initialization cost again.
Because the agent calls TTS repeatedly, the initialization overhead would be included in every measurement, obscuring the actual synthesis performance.

### Wrapping Kokoro in a server

`utils/kokoro_server.py` is a lightweight FastAPI + Uvicorn wrapper around the Kokoro library. It turns the library into a long-running service, avoiding repeated model initialization overhead.

### Install Kokoro and start the server

The cell below installs the `kokoro` package and the server's web dependencies, then the next cell starts the server. It is safe to re-run.

```python
!python3 -m pip install -q kokoro soundfile fastapi uvicorn
!python3 -c "import kokoro; print('kokoro', kokoro.__version__)"
```

Now start the server on port 8092 using `utils/start_kokoro_server.sh`. It
loads the model into VRAM once and keeps it resident, so later synthesis calls do
not have to pay the model initialization overhead.

```python
!bash utils/start_kokoro_server.sh
```

### The custom tool: `kokoro_tts`

`kokoro_tts` is a custom tool added to Hermes. It sends text to the local Kokoro
server and returns spoken audio as a `WAV` file (uncompressed, preserving the
original quality). Hermes is extensible: you can add custom tools and skills to it. The `kokoro_tts` tool is an example of this.

The cell below installs the tool into Hermes. The tool must live inside the Hermes
`tools` package, because the cell below performs `from tools.registry import ...`. Copying
it anywhere else (`~/.hermes/tools/`, for example) leaves the
tool unimportable. The agent then loops without ever calling it, with no error
message to explain the issue. The code locates the real package and verifies that the tool
is registered, rather than assuming that copying the files is sufficient.

```python
%%bash
set -uo pipefail

SRC="./custom_tools/kokoro_tts_tool.py"
if [ ! -f "$SRC" ]; then
    echo "[ERROR] $SRC not found. Run this from the repository root."
    exit 1
fi

# Resolve the Hermes installation, whichever layout this machine uses:
# a per-user install (~/.hermes/hermes-agent) or a system one (/usr/local).
# Require both tools/ and venv/bin/python, since the verification step below
# runs that interpreter.
HERMES_ROOT=""
for cand in "$HOME/.hermes/hermes-agent" /usr/local/lib/hermes-agent; do
    if [ -x "$cand/venv/bin/python" ] && [ -d "$cand/tools" ]; then
        HERMES_ROOT="$cand"
        break
    fi
done

if [ -z "$HERMES_ROOT" ]; then
    echo "[ERROR] Could not find a complete Hermes install (tools/ + venv/)."
    echo "        Has hermes finished installing?"
    exit 1
fi

echo "[INFO] Deploying custom Kokoro TTS tool to $HERMES_ROOT/tools ..."
cp "$SRC" "$HERMES_ROOT/tools/"
echo "[OK] Copied kokoro_tts_tool.py -> $HERMES_ROOT/tools/"

# Verify the tool registers, not just that the file copied.
"$HERMES_ROOT/venv/bin/python" -c "
import sys; sys.path.insert(0, '$HERMES_ROOT')
from tools.registry import registry
import tools.kokoro_tts_tool
names = getattr(registry, 'tools', None) or getattr(registry, '_tools', {})
assert 'kokoro_tts' in names, f'kokoro_tts NOT registered; found {sorted(names)}'
print('[OK] kokoro_tts is registered and callable by the agent.')
"
```

#### Parameters of the `kokoro_tts` tool

| Parameter | Purpose |
| :--- | :--- |
| `text` | The full text to synthesize, in one call. |
| `text_file` | Path to a UTF-8 file to synthesize. Preferred for long text, so the agent passes a path instead of the whole passage inline. |
| `mode` | The inference strategy, `sequential` or `batched`. `sequential` is the native Kokoro baseline; `batched` is the optimization we will apply in [Step 5](#step-5-optimize-the-tool-with-batching). This parameter is what lets us compare the two modes. The default is `sequential`. |
| `voice` | Voice name (default `af_heart`). |
| `batch_size` | Sentences per GPU forward pass when `mode` is `batched` (default `24`). Ignored in `sequential` mode. |
| `output_path` | Where to save the WAV (default is `~/.hermes/audio_cache/`). |

The tool is defined in `custom_tools/kokoro_tts_tool.py` and backed by
`utils/kokoro_server.py`. The cell below runs the tool on the input text and profiles how it performs.

```python
!hermes chat --yolo --oneshot -q "Use the kokoro_tts tool and pass the file path 'input_text.txt' as the text_file parameter to convert the text to speech. Do not read the file yourself"
```

The cell below shows the dashboard's **Overview** tab and generates links to the full dashboard and the MLflow trace view.

```python
import importlib, sys, os, logging, warnings

sys.path.insert(0, os.path.abspath("utils"))

logging.disable(logging.WARNING)   # mute Streamlit's "missing ScriptRunContext" etc.
warnings.filterwarnings("ignore")  # mute plotly/mlflow UserWarning/FutureWarning etc.

import hermes_profiler
importlib.reload(hermes_profiler)          # pick up edits without a kernel restart

SESSION_ID = None   # a run ID, or None to auto-fetch the latest session
hermes_profiler.show_session_overview(SESSION_ID)

logging.disable(logging.NOTSET)
```

> **Note:** The very first Kokoro run compiles GPU kernels for your input shape, which can take a few minutes. Later runs reuse the compiled kernels and are faster.

### How `kokoro_tts` works, and why the first run is slow

`kokoro_tts` runs the Kokoro TTS model locally on the AMD MI300X GPU. In its default
mode it processes the input one sentence at a time and saves the result as a WAV
file.

Load this run in the dashboard the same way as before: click **Fetch**, select
the run at the top of the list, then click **Load / Reload**. You should see:

- The `kokoro_tts` span occupies the largest part of the timeline, so it is the
  primary contributor to end-to-end latency.
- GPU utilization stays low for most of the run. The MI300X is not being fully
  utilized.
- Because sentences are processed one at a time, each inference gives the GPU only a
  small amount of work, which is inefficient.
- As the number of sentences grows, the tool runs more inference passes, so latency
  grows **almost linearly**.

So, `kokoro_tts` works and produces the correct audio, but it clearly underutilizes the GPU.
That is the bottleneck the dashboard reveals, which provides the motivation to optimize the
tool, which is the next step.

## Step 5: Optimize the tool with batching

The dashboard showed the bottleneck: the tool feeds the GPU one sentence at a
time. To address this, `kokoro_tts` was extended with an optimized `batched` mode by `utils/kokoro_server.py`.

Instead of processing texts one sentence at a time (the `sequential` mode),
the new `batched` mode groups many sentences together and sends them to the GPU in a single 
forward pass, giving the hardware much more work to do at once. The following diagram illustrates the difference between the two modes:

<p align="center">
<img alt="Comparison of sequential and batched processing. In sequential mode five sentences s1 to s5 each take their own forward pass, causing five GPU launches and low utilization. In batched mode the same five sentences are grouped and length-bucketed into a single GPU launch with high utilization." src="../assets/embedded-156aea7e8892e55b8c96.png" width="94%">
</p>

The cell below re-runs the exact same input, this time asking for `mode='batched'` for comparison.

```python
!hermes chat --yolo --oneshot -q "Use the kokoro_tts tool with the mode parameter set to 'batched', and pass the file path 'input_text.txt' as the text_file parameter. Do not read the file yourself"
```

The cell below shows the dashboard's **Overview** tab for the re-run and generates links to the full dashboard and the MLflow trace view.

```python
# This cell shows an overview tab alone; for a detailed view, click on the links provided when running this cell.
import importlib, sys, os, logging
sys.path.insert(0, os.path.abspath("utils"))
logging.disable(logging.WARNING)          # mute Streamlit's import-time warnings
import hermes_profiler
importlib.reload(hermes_profiler)          # pick up edits without a kernel restart
logging.disable(logging.NOTSET)

SESSION_ID = None   # a run ID, or None to auto-fetch the latest session
hermes_profiler.show_session_overview(SESSION_ID)
```

Load this run in the dashboard the same way as before: click **Fetch**, select
this run (it appears at the top), then click **Load / Reload**. Then compare it
side by side with the sequential run.

### What changed under the hood

Kokoro does not support native batching, so `mode='batched'` is a reimplementation
of the model's forward pass. Three ideas are involved:

- **Batching:** Multiple sentences are grouped, so the GPU processes more work per
  forward pass, improving hardware utilization.
- **Length bucketing:** Sentences of similar length are grouped into the same batch,
  reducing wasted padding.
- **Correct batched processing:** Padding and attention masks keep each sentence
  independent, and the final audio is trimmed back to its true length.
  
Three functions in `utils/kokoro_server.py` implement the ideas:

| Function | Responsibility |
|---|---|
| `KokoroEngine._run_batched()` | The optimized runner, used in place of the sequential loop. It sorts sentences by token length (bucketing), chunks them into batches of `batch_size`, pads each batch into one tensor, and then extracts the generated audio back into individual sentence outputs. |
| `batched_forward()` | The replacement forward pass, standing in for `KModel.forward_with_tokens`. It builds a real padding mask from true sequence lengths, packs the duration LSTM, zeroes padded durations, and constructs a per-sentence alignment matrix. |
| `_batched_f0n_train()` | Batch-safe replacement for the model's `F0Ntrain`. It packs the shared LSTM by frame count so padding in one sentence cannot leak into the next. |

### What you should see in the dashboard

- With `mode='batched'`, `kokoro_tts` completes significantly faster than in 
  sequential mode.
- The GPU does more work per forward pass, so utilization is higher and overhead
  is lower.
- Fewer GPU launches overall, because sentences are processed together rather than
  one at a time.
- CPU activity stays relatively low, confirming the workload is GPU-bound during
  synthesis.
- The audio output is the same; only the execution time drops.

> **Takeaway:** Batching improves throughput and GPU utilization, which makes it the
> preferred mode for longer inputs.

## Step 6: Visualize the improvement

The chart makes the progression clear. Tool execution time affects overall performance in all
three approaches, not just when comparing sequential and batched execution. Moving from cloud-based Edge TTS to
the local Kokoro model and then to batched Kokoro overcomes the drawbacks of the cloud-based approach and delivers better performance while keeping all processing on local hardware.

<p align="center">
<img alt="Three approaches compared as cards. Edge TTS is the cloud baseline with zero setup, a 5,000-character cap and text leaving the machine. Kokoro sequential is the local baseline, one sentence per GPU pass, correct but under-using the GPU. Kokoro batched is the local optimized approach with many sentences per GPU pass, length bucketing and masks, same audio at far higher throughput." src="../assets/embedded-b9a0cbe8b09a7f50c808.png" width="94%">
</p>

The cell below uses **Matplotlib** to plot the tool execution time of the
three approaches side by side, so the cloud-to-local move and the
sequential-to-batched optimization show up in a single view.

> **Use your own numbers:** `edge_time`, `seq_time`, and `batched_time` are
> pre-filled with example values so the chart renders meaningfully even if you have not 
> made any runs. Replace the data with the execution seconds from your own runs (each
> tool's output line and the profiling dashboard). Note that for long inputs, Edge
> truncates its output, so the execution time shown is just to provide context for the cloud-based 
> baseline rather than as a direct comparison with the other approaches.

```python
%matplotlib inline

import os
import matplotlib.pyplot as plt
from matplotlib import font_manager

# --- Tool execution time (seconds) for each approach ---
# Default values; replace them with the execution seconds from your own runs,
# taken from each tool's output line and the profiling dashboard.
edge_time = 11.1      # Edge TTS (cloud) - note: truncates long text (~5 min cap)
seq_time = 83.8      # Kokoro, sequential mode (local, unoptimized)
batched_time = 7.0    # Kokoro, batched mode (local, optimized)

for _f in ("Arial", "Liberation Sans", "DejaVu Sans"):
    if any(_f in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break

AMD_RED = "#ED1C24"
INK     = "#1A1A1A"
SUBINK  = "#5B6270"
EDGE_C  = "#F08418"   # cloud Edge (dashboard orange)
SEQ_C   = "#2E6DB4"   # local sequential (dashboard blue)
BATCH_C = "#1EAAB4"   # local batched (teal)

labels = ["Edge\n(cloud)", "Sequential\n(local)", "Batched\n(local, optimized)"]
times  = [edge_time, seq_time, batched_time]
colors = [EDGE_C, SEQ_C, BATCH_C]

fig, ax = plt.subplots(figsize=(7.6, 4.8))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

bars = ax.bar(labels, times, color=colors, width=0.62, zorder=3,
              edgecolor="white", linewidth=1.2)
ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
ax.set_ylabel("Tool execution time (seconds)", fontsize=12, color=INK)
ax.set_title("TTS tool execution time: cloud Edge vs local Kokoro",
             fontsize=14, fontweight="bold", color=INK, pad=30)
ax.text(0.5, 1.045, "lower is better", transform=ax.transAxes, ha="center",
        va="bottom", fontsize=10.5, color=SUBINK, style="italic")
ax.set_ylim(0, max(times) * 1.32)
ax.bar_label(bars, fmt="%.1f s", padding=4, fontweight="bold",
             fontsize=12, color=INK)
ax.tick_params(colors=INK, labelsize=11)

notes = []
if seq_time and batched_time:
    notes.append(f"batched is {seq_time / batched_time:.1f}x faster than sequential")
if edge_time and batched_time:
    notes.append(f"batched is {edge_time / batched_time:.1f}x faster than Edge")
if notes:
    ax.text(0.98, 0.96, "\n".join(notes), transform=ax.transAxes,
            ha="right", va="top", fontsize=11.5, fontweight="bold",
            color=AMD_RED,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FDECEC",
                      edgecolor=AMD_RED, linewidth=1.0))

for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color("#C9CDD6")

plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/tts_execution_comparison.png", dpi=150, bbox_inches="tight",
            facecolor="white")
plt.show()

print(f"Edge: {edge_time:.1f}s | Sequential: {seq_time:.1f}s | "
      f"Batched: {batched_time:.1f}s")
```

## Applying the profiling loop to other workloads

You have now run the full workflow for a single example. The same observability-driven
workflow applies to a wide range of AI workloads beyond TTS.

<p align="center">
<img alt="The observability-driven loop as five numbered steps: 1 Run the agent, 2 Fetch in the dashboard, 3 Inspect spans and GPU, 4 Optimize the slow tool, 5 Measure again, with a dashed arrow looping back to step 1 to repeat until the bottleneck is gone." src="../assets/embedded-2bc01d119764ce83f43e.png" width="92%">
</p>

**Try different inputs for this TTS example**

1. Regenerate `input_text.txt` by re-running the [Prepare your input text](#6-prepare-your-input-text) cell with
   a new topic, or edit the `input_text.txt` file directly. Longer, multi-sentence texts make the
   speed-up from batching more apparent.
2. Re-run the two synthesis cells: `kokoro_tts` in the default (sequential) mode in [Step 4](#step-4-local-tts-with-kokoro),
   then with `mode='batched'` in [Step 5](#step-5-optimize-the-tool-with-batching).
3. In the dashboard, click **Fetch**, select the latest run, click
   **Load / Reload**, and then update the Matplotlib cell in [Step 6](#step-6-visualize-the-improvement) with your own execution
   times to compare.

> **Re-running the Edge baseline:** Once `kokoro_tts` is installed, the agent will
> usually prefer it, so the cell in [Step 1](#step-1-establish-a-baseline-with-edge-tts) no longer measures the performance of 
> Edge. To get an Edge run again, remove the custom tool first before re-running the cell:
>
> ```bash
> rm -f ~/.hermes/hermes-agent/tools/kokoro_tts_tool.py       # per-user install
> rm -f /usr/local/lib/hermes-agent/tools/kokoro_tts_tool.py  # container image
> ```
>
> To switch back to `kokoro_tts`, re-run the custom tool install cell in [Step 4](#step-4-local-tts-with-kokoro).

**Try it on your own use cases**

Text-to-speech was only an example. Point the same workflow at **any** Hermes task: a
research query, a coding task, a multi-tool workflow, to:

- **Profile the run** in the dashboard (click **Fetch**, select the run, then
  **Load / Reload**) to see the CPU and GPU timelines, and which tool dominated.
- **Inspect the MLflow traces** (the **Traces** view) to drill into each session,
  LLM call, and tool call and see exactly where time was spent.
- **Find the bottleneck, optimize it, and measure again**, following the same loop you went through here.

> **Tip.** The bigger the workload, the clearer the optimization. Let the dashboard and
> traces eliminate the guesswork and point you to the slow step.

## Cleanup

This workshop writes a few artifacts, including audio files, input text, and telemetry data. 
Also, the container remains running until it is stopped. 
Follow the steps below to clean up after completing the tutorial:

- Delete generated files in the working directory:

    ```bash
    rm -f input_text.txt output_audio.mp3  # if Edge TTS produced multiple audio files, remove them manually
    rm -rf outputs
    ```

- Stop and remove the container:

    ```bash
    docker stop amd-agentic-ai-profiling
    docker rm amd-agentic-ai-profiling
    ```

- Optionally, delete the container image to reclaim disk space if you do not plan to run it again:

    ```bash
    docker rmi vllm/vllm-openai-rocm
    ```

With these steps, the Hugging Face cache in `$HOME/hf_cache` is kept so future runs do not re-download
the models. Remove the models too if you want to reclaim the disk space:

```bash
rm -rf "$HOME/hf_cache"
```

## Conclusion and key takeaways

This exercise walked through an observability-driven optimization workflow that
compared cloud-based Edge TTS with a local Kokoro TTS, identified the bottleneck
through telemetry, optimized the local implementation, and measured the improvement.

| Phase | Tool | Execution | Outcome |
| :--- | :--- | :--- | :--- |
| **1 &middot; Edge TTS** | `text_to_speech` | Cloud-based | Convenient for short inputs, but sends the text off-machine and has higher latency than a local model. |
| **2 &middot; Kokoro, sequential** | `kokoro_tts` | Local MI300X, one sentence/pass | Complete output, but slower with low GPU utilization. |
| **3 &middot; Kokoro, batched** | `kokoro_tts` (`mode='batched'`) | Local MI300X, many sentences/pass | Same output, significantly better execution time and GPU utilization. |

### What we learned

- **Cloud to local:** Moving from Edge TTS to a local Kokoro model removed the
  input-length limit and kept text processing on the machine.
- **Establish a baseline:** The sequential Kokoro implementation established a baseline
  for local TTS performance.
- **Measure, do not guess:** The observability dashboard made it clear that the
  local model was under-utilizing the GPU.
- **Optimize with batching:** A batched mode processes multiple sentences together,
  cutting inference and launch overhead and using the GPU better.
- **Validate the improvement:** Re-running the workflow and profiling confirmed that the batched Kokoro mode beat the sequential mode by a wide margin and is a significant improvement over Edge TTS.
- **Iterate:** Switch to a local model, establish a baseline, find the bottleneck,
  optimize, and measure again.

## Diving deeper

This section offers optional experiments and background information for users who want to explore further.

### Extra experiments

- **Raw MLflow UI (advanced):** Browse everything directly at
  `http://<server-ip>:5004`, under the **Traces** and **Runs** tabs.
- **Analysis tab:** Click **Analyze with Hermes** in the dashboard for automatic,
  plain-language suggestions based on the run's `tool_breakdown.csv`.

### Understanding the Kokoro kernel cache

When Kokoro runs, it compiles GPU kernels for the specific input shapes it encounters. The following sections explain how this compilation process works and how the kernel cache affects performance.

#### Kernel compilation and reuse

For a specific input to Kokoro:

- **First run (cold run):** The first execution compiles the required GPU kernels.
  This compilation overhead increases execution time.
- **Subsequent runs:** From the second run onward, compiled kernels are reused from
  the cache, giving faster execution and lower latency.
- **Clearing the cache:** For Python-based executions, the kernel cache can be
  cleared with `utils/clear_cache.sh`. In this workshop, however, Kokoro runs as a
  persistent server, so clearing the cache means stopping the server, deleting the
  cache, and restarting before benchmarking again.

##### Understanding the cache

MIOpen stores compiled and tuned GPU kernels in its cache. With
`MIOPEN_FIND_MODE=FAST`, MIOpen skips the expensive kernel search on later runs and
reuses the best cached kernel. The Code Object Manager (COMGR) caches the LLVM compilation artifacts, 
avoiding recompilation of GPU code on future executions.

##### Is the cache reused for all subsequent input?

Not always. The cache is built for specific kernel configurations that depend on
factors such as input shape, including sequence length and tensor dimensions. If a new input
needs a configuration that has not been compiled before, MIOpen compiles and caches
it during that execution. Once the compiled kernel is cached, later runs with the same configuration reuse
it.

### Troubleshooting

A few issues might come up, most likely around permissions and accessing the services
from a remote browser.

**Model downloads or Kokoro install fail with a permission error:** This happens
when `~/.cache/huggingface` was created by a root process and you cannot
`sudo chown` it back to yourself (common on shared GPU clusters). Use the container start
command above with a fresh, user-owned cache directory:
`-v "$HOME/hf_cache:/root/.cache/huggingface"`. Docker creates `$HOME/hf_cache`
owned by you, so downloads work without any `chown`.

**The dashboard or MLflow link does not load on my machine:** The links point at
the host running the backend. If that is a remote server:

- Forward the ports over SSH and use the `localhost` links:
  `ssh -L 8501:localhost:8501 -L 5004:localhost:5004 <user>@<server>`, then set
  `os.environ["HERMES_PROXY_BASE"] = ""` in the first cell under [Launching the profiling dashboard](#launching-the-profiling-dashboard) before running it.
- Or, if ports `8501`/`5004` are directly reachable on your network, use the
  server-IP links the cell prints.
- On the AMD hosted-notebook proxy, use the proxied links exactly as printed and
  leave `HERMES_PROXY_BASE` unset.

**`which hermes` prints nothing:** The JupyterLab kernel may not have inherited the
login shell's `PATH`. Re-run [Step 5: Check your setup](#5-check-your-setup), which adds both
`/usr/local/bin` and `~/.local/bin` to `PATH`. If the notebook still cannot find Hermes,
make sure the backend has finished starting.

## Further reading

For deeper server reference and troubleshooting notes, see the `README` of the workshop repo at 
<https://github.com/shailensobhee/amd-agentic-ai-profiling-workshop>.
