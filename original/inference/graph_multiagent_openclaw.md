> [Upstream source](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/inference/graph_multiagent_openclaw.ipynb) · Snapshot: `0071bfd4bd3bd9525405c09a6ae69c983b09312a`

# Multi-agent incident triage with OpenClaw

**Author:** Kalyan Archakam  
**Knowledge level:** Intermediate  
**Publication date:** August 20, 2026 

This tutorial shows how to build a multi-agent incident triage system that runs on a single
AMD Instinct™ MI300X GPU using [OpenClaw](https://docs.openclaw.ai/) and [vLLM](https://docs.vllm.ai/).
You will serve a local Qwen3.6-35B-A3B model, configure a graph with three specialist agents, and run them
against real observability data from a microservices fault injection study.

## From prompt engineering to graph engineering

Over time, the way practitioners structure AI workloads has evolved through several stages:

**Prompt engineering**: A single model call, shaped by carefully crafted input text. The model is a black box, and the lever is your input.

**Context engineering**: The model call is the same as prompt engineering, but the surrounding context is curated:
retrieval-augmented generation (RAG), system prompts, tool descriptions, few-shot examples, etc. Quality
of the context window determines quality of the output.

**Loop engineering**: A single agent runs a plan → act → observe cycle, retrying until a
termination condition is met. This is the default pattern for most AI coding agents today.

**Graph engineering**: The loop is replaced by an explicit graph. Multiple agents run in
parallel on different sub-problems, with their outputs converging to a coordinator that makes a
decision. The structure between nodes (for example, which agent sees which data, and in what order) becomes
as important as the agents themselves.

The shift from loop to graph engineering is not about adding complexity. It is about matching the approach with the structure
of the problem. Incident triage is naturally parallel: logs, metrics, and traces are independent
signals that can be read simultaneously. A loop processes them sequentially and asks one agent
to hold all three in context at once. On the other hand, a graph assigns each signal to a specialist and uses a
coordinator to correlate the results, which is similar to how an experienced site reliability engineering (SRE) team usually works.

## What you'll build

This figure illustrates the graph engineering approach taken by this tutorial for multi-agent triaging.

<img src="../assets/embedded-42095da5a8df0dce881b.png" alt="Fan-out/fan-in multi-agent triage graph: a coordinator spawns logs, metrics and traces specialists that each write findings to their own workspace, then correlates them into a root-cause report" width="820"/>

**How the graph maps onto OpenClaw:**

| Graph element | OpenClaw primitive | Configured in |
|---|---|---|
| Nodes | `openclaw agents`, one workspace per specialist | [Part 5.1](#51-add-the-graph-nodes) |
| Typed edges | Each agent's `AGENTS.md` briefing | [Part 5.2](#52-define-the-typed-edges) |
| Fan-out | `sessions_spawn` × 3, before the first yield | [Part 5.4](#54-define-the-control-flow) |
| Fan-in | `sessions_yield` + per-agent `findings.md` | [Part 5.4](#54-define-the-control-flow) |
| Adjacency list | `agents.defaults.subagents.allowAgents` | [Part 5.3](#53-define-the-adjacency-list) |
| Bounded cycle | One-retry rule in the coordinator protocol | [Part 5.4](#54-define-the-control-flow) |

Here is a summary of the steps this tutorial will guide you through:
- Make sure all prerequisites for this project are satisfied ([Prerequisites](#prerequisites)).
- Get vLLM and OpenClaw ready ([Part 1](#part-1-start-the-vllm-server) to [Part 3](#part-3-connect-openclaw-to-the-model-and-start-the-gateway)).
- Fetch incident data to be triaged, and curate the inputs for the different agents ([Part 4](#part-4-fetch-the-incident-data)).
- Configure the fan-out/fan-in multi-agent triage graph ([Part 5](#part-5-configure-the-graph)).
- Run the triage graph ([Part 6](#part-6-run-the-triage-in-the-tui)).
- Clean up after the exercise ([Part 7](#part-7-summary-and-cleanup)).

## Prerequisites

This tutorial was developed and tested using the following setup.

### Operating system

Ubuntu 22.04 or 24.04.

### Hardware

**AMD Instinct GPUs**: This tutorial was tested on a single AMD Instinct MI300X GPU (192 GB HBM3). Ensure you are using an AMD Instinct GPU or compatible hardware with ROCm support and that your system meets the [official requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

You will also need approximately **70 GB of free disk space** for the model weights.

### Software

**ROCm 7.0 or later**: Install and verify ROCm by following the [ROCm install guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html). This tutorial was tested on ROCm 7.2.3. After installation, confirm your setup using:

```bash
amd-smi
```

This command lists your AMD GPUs with relevant details.

**Docker**: Ensure Docker is installed and configured correctly. Follow the [Docker installation guide](https://docs.docker.com/engine/install/) for your operating system.

> **Note**: Ensure Docker permissions are correctly configured. To allow non-root access, run:
>
> ```bash
> sudo usermod -aG docker $USER
> newgrp docker
> ```
>
> Verify Docker is working correctly:
>
> ```bash
> docker run hello-world
> ```

**Install and launch JupyterLab:** You run JupyterLab on the host (the droplet). The model runs in a container that you start from a
notebook cell in [Part 1](#part-1-start-the-vllm-server), and OpenClaw is installed on the host in [Part 2](#part-2-ensure-openclaw-is-installed). The notebook, the OpenClaw gateway, and the vLLM server all run on `localhost`, so they can communicate over `127.0.0.1`.

In the droplet terminal, create a virtual environment and install JupyterLab:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install jupyterlab
```

Start the Jupyter server in the same terminal:

```bash
jupyter-lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

> **Note**: Ensure port `8888` is not already in use. If it is, replace `--port=8888` with another
> port, for example `--port=8890`.

After running the `jupyter-lab` command, click the link in the terminal to open JupyterLab. The
link has the form `http://127.0.0.1:<PORT>/lab?token=<TOKEN_VALUE>`.

> **Note**: Make sure this notebook is in the directory you launched Jupyter from, or upload it
> after Jupyter starts. You can download this notebook from the [AI Developer Hub GitHub repository](https://github.com/ROCm/gpuaidev).

Run the remaining steps in this tutorial **from inside this notebook (or a JupyterLab terminal)** after the Jupyter server is up.

## Part 1: Start the vLLM server

Run the cell below to start a Docker container that serves the Qwen3.6-35B-A3B model with the
latest vLLM build for ROCm. `vllm/vllm-openai-rocm` uses `vllm serve` as its entrypoint, so the
model and serving flags are passed as arguments after the image name. The `-v $(pwd)/hf-cache:/root/.cache/huggingface` flag mounts a local directory over the container's Hugging Face cache so the downloaded weights persist across container restarts.

`vllm/vllm-openai-rocm:latest` is the latest official vLLM image built for ROCm and includes support for
the Qwen3.6 architecture (`Qwen3_5MoeForCausalLM`). The `rocm/vllm` image used in some other
tutorials predates this architecture and does not support it.

> **Note**: Replace `abc-123` below with a secure, unique API key.

```python
%%bash
docker run -d \
    --name vllm-server \
    --ipc=host \
    --privileged \
    --device=/dev/kfd \
    --device=/dev/dri \
    -p 8000:8000 \
    -v $(pwd)/hf-cache:/root/.cache/huggingface \
    vllm/vllm-openai-rocm:latest \
    Qwen/Qwen3.6-35B-A3B \
    --served-model-name Qwen3.6-35B-A3B \
    --host 0.0.0.0 --port 8000 \
    --api-key abc-123 \
    --max-model-len 32768 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --trust-remote-code
```

Verify the server is running by executing the cell below. It might take a few minutes to load the
model the first time. You can also watch the progress in a terminal with
`docker logs -f vllm-server` and look for `Application startup complete`.

```python
import urllib.request, json, time

# Poll the server until it is ready (up to 5 minutes)
print("Waiting for server to be ready", end="", flush=True)
deadline = time.time() + 300
ready = False
while time.time() < deadline:
    try:
        req = urllib.request.Request(
            "http://localhost:8000/v1/models",
            headers={"Authorization": "Bearer abc-123"},
        )
        with urllib.request.urlopen(req, timeout=3) as r:
            models = json.loads(r.read())
        ready = True
        break
    except Exception:
        print(".", end="", flush=True)
        time.sleep(5)

if ready:
    print("\n✅ Server is ready")
    for m in models.get("data", []):
        print(f"   Model: {m['id']}")
else:
    print("\n❌ Server did not become ready within 5 minutes")
    print("   Check logs with: docker logs vllm-server")
```

**Troubleshooting**

If the server does not become ready, check these common issues:

- **Container name already in use**: A container from a previous run still exists. Remove it and
  re-run the launch cell:

  ```bash
  docker rm -f vllm-server
  ```

- **Out of memory (OOM)**: The model needs roughly 70 GB of GPU memory. Check free memory with this command, and free up some memory if needed by closing other GPU-intensive applications:

  ```bash
  amd-smi monitor
  ```

- **Startup errors:** Inspect the container logs:

  ```bash
  docker logs vllm-server
  ```

## Part 2: Ensure OpenClaw is installed

OpenClaw runs your agents as a local service called the **gateway**. If the `openclaw` CLI is already present, you do **not** need to install anything. Only run the installer if OpenClaw is missing. Run the cell below first for a check:

```python
!command -v openclaw >/dev/null 2>&1 && openclaw --version \
  || echo "OpenClaw not found - install it in a Jupyter terminal (see above), then re-run this cell."
```

If it prints a version, OpenClaw is ready. Continue to [Part 3](#part-3-connect-openclaw-to-the-model-and-start-the-gateway).

If it prints `OpenClaw not found`, open a **Jupyter terminal** (click **Terminal** in the **Launcher**) and install OpenClaw there. The installer runs a short interactive **doctor** step, so it needs a real terminal and can't be run from a notebook cell:

  ```bash
  curl -fsSL --proto '=https' --tlsv1.2 https://openclaw.ai/install.sh | bash
  ```

Accept the recommended configuration repairs when prompted, then re-run the check above to verify that the installation has been successful.

## Part 3: Connect OpenClaw to the model and start the gateway

This part connects OpenClaw to the running vLLM server and starts the gateway process.

### 3.1 Connect OpenClaw to the vLLM endpoint

Onboard the local vLLM endpoint by registering it with OpenClaw and associating it with a model. The model id and the api-key must match the values specified for `--served-model-name` and `--api-key`, respectively, from [Part 1](#part-1-start-the-vllm-server).

```python
%%bash
openclaw onboard --non-interactive --mode local \
  --auth-choice custom-api-key \
  --custom-base-url "http://localhost:8000/v1" \
  --custom-model-id "Qwen3.6-35B-A3B" --custom-provider-id "vllm" \
  --custom-compatibility openai --custom-api-key "abc-123" \
  --secret-input-mode plaintext \
  --skip-channels --skip-search --skip-hooks --skip-ui --skip-health --accept-risk
```

### 3.2 Keep the full tool surface (lean mode off)

Leave **lean mode off** for this graph. Lean mode shrinks the tool surface and moves tools such as 
`sessions_spawn` and `sessions_yield` behind a `tool_search` step. A small local model might fail to discover the tools then, preventing the coordinator from fanning out and resulting in an error such as `sessions_spawn tool is not available`. Keeping the full surface exposed allows `main` to call the spawn and yield tools directly.

```python
%%bash
openclaw config set agents.defaults.experimental.localModelLean false
echo "lean mode off (sessions_spawn/yield stay directly available)"
```

### 3.3 Start the gateway in the background

This container has no `systemd`, so run the gateway directly with `nohup`.

```python
%%bash
pkill -f openclaw-gateway 2>/dev/null || true
sleep 1
nohup openclaw gateway run > ~/gateway.log 2>&1 &
sleep 3
tail -n 8 ~/gateway.log
```

### 3.4 Test the model

Perform a smoke test by running this command:

```python
!openclaw agent --local --session-id smoke --message "Reply with exactly: pong" --thinking off
```

It should reply `pong`.

## Part 4: Fetch the incident data

This tutorial uses one fault window from the
[Nezha](https://github.com/IntelligentDDS/Nezha) observability dataset: real logs, metrics,
and traces captured from the **OnlineBoutique** microservices application during a controlled
fault injection experiment. A fault was injected into exactly one service. All three signals
carry a trace of it, but no single signal is sufficient to identify the cause.

The raw per-minute CSV files are 4–15 MB each. The cells below first configure the fault window
and a download helper, then curate each signal (metrics, traces, and logs) into compact, model-readable files in `/workspace/incident/`.

### 4.1 Configure the fault window and download helper

First, pin the incident to a specific date and a specific three-minute fault window in the Nezha dataset,
list the ten OnlineBoutique pods, and define a small `fetch()` helper for pulling a raw CSV file from
GitHub. This cell only sets things up and doesn't download or write any data.

```python
import os, io, csv, json, urllib.request

NEZHA_DATE   = "2022-08-22"
FAULT_WINDOW = ["04:44", "04:45", "04:46"]   # fault injected at 04:44:16
BASELINE_MIN = "04_36"
FAULT_MIN    = "04_45"
INCIDENT_DIR = "/workspace/incident"
RAW = f"https://raw.githubusercontent.com/IntelligentDDS/Nezha/main/rca_data/{NEZHA_DATE}/"

PODS = [
    "adservice-5f6585d649-fnmft",        "cartservice-579f59597d-wc2lz",
    "checkoutservice-578fcf4766-9csqn",  "currencyservice-cf787dd48-vpjrd",
    "emailservice-55fdc5b988-f6xth",     "frontend-579b9bff58-t2dbm",
    "paymentservice-9cdb6588f-554sm",    "productcatalogservice-668d5f85fb-wckp8",
    "recommendationservice-6cfdd55578-gfj6q", "shippingservice-7b598fc7d-lmggd",
]
svc = lambda pod: pod.split("-")[0]
os.makedirs(INCIDENT_DIR, exist_ok=True)

def fetch(path):
    req = urllib.request.Request(RAW + path, headers={"User-Agent": "notebook"})
    return urllib.request.urlopen(req, timeout=180).read().decode("utf-8", "replace")
```

### 4.2 Curate the metrics signal

Download each service's per-minute metric CSV file, keeping only the rows inside the fault window, and
extracting the CPU usage, server-latency P95, and workload. The curated result is written to
`/workspace/incident/metrics.csv` and the cell prints the file's size when it finishes.

```python
# Metrics: CPU usage, server-latency 95th-percentile (P95), and workload for each service across the fault window.
rows = []
for pod in PODS:
    for r in csv.DictReader(io.StringIO(fetch(f"metric/{pod}_metric.csv"))):
        if r["Time"][11:16] in FAULT_WINDOW:
            rows.append([svc(pod), r["Time"][11:19],
                         round(float(r["CpuUsageRate(%)"]), 1),
                         round(float(r["PodServerLatencyP95(s)"]), 3),
                         round(float(r["PodWorkload(Ops)"]), 1)])
with open(f"{INCIDENT_DIR}/metrics.csv", "w") as f:
    f.write("service,time,cpu_usage_rate_pct,server_latency_p95_s,workload_ops\n")
    f.write("\n".join(",".join(map(str, r)) for r in rows) + "\n")
print(f"metrics.csv  {os.path.getsize(f'{INCIDENT_DIR}/metrics.csv')} bytes")
```

### 4.3 Curate the traces signal

Download the trace CSV files for the fault minute and a baseline minute captured eight minutes
earlier, compute the span-latency P95 per service, and write the per-service slowdown to
`/workspace/incident/traces.csv`.

```python
# Traces: span-latency P95 per service: fault minute vs a baseline minute.
def p95(vals): return sorted(vals)[int(len(vals) * 0.95)] if vals else 0

def p95_by_svc(minute):
    d = {}
    for r in csv.DictReader(io.StringIO(fetch(f"trace/{minute}_trace.csv"))):
        d.setdefault(svc(r["PodName"]), []).append(int(r["Duration"]))
    return {s: p95(v) for s, v in d.items()}

base, fault = p95_by_svc(BASELINE_MIN), p95_by_svc(FAULT_MIN)
with open(f"{INCIDENT_DIR}/traces.csv", "w") as f:
    f.write("service,baseline_p95_ms,fault_p95_ms,slowdown_x\n")
    for s in sorted(set(base) | set(fault)):
        b, x = base.get(s, 0) / 1e6, fault.get(s, 0) / 1e6
        f.write(f"{s},{round(b,2)},{round(x,2)},{round(x/b,1) if b else 0}\n")
print(f"traces.csv   {os.path.getsize(f'{INCIDENT_DIR}/traces.csv')} bytes")
```

### 4.4 Curate the logs signal

Finally, download the fault-minute log CSV file and summarize it per service, including the line count,
the error and warning count, and a sample message. The summary is written to
`/workspace/incident/logs.txt`.

```python
# Logs: per-service line count, error/warning count, and one sample message.
counts, errors, sample = {}, {}, {}
for r in csv.DictReader(io.StringIO(fetch(f"log/{FAULT_MIN}_log.csv"))):
    s = svc(r["PodName"]); counts[s] = counts.get(s, 0) + 1
    if any(k in r["Log"].lower() for k in ('"error"','"warn"',"exception","failed","panic")):
        errors[s] = errors.get(s, 0) + 1
    if s not in sample:
        try:    sample[s] = json.loads(json.loads(r["Log"])["log"])["message"][:100]
        except: sample[s] = r["Log"][:100]
with open(f"{INCIDENT_DIR}/logs.txt", "w") as f:
    f.write(f"# OnlineBoutique logs: fault minute {FAULT_MIN.replace('_',':')}\n")
    f.write("# service: lines | errors/warnings | sample\n\n")
    for s in sorted(counts, key=lambda k: -counts[k]):
        f.write(f"{s}: {counts[s]} lines | {errors.get(s,0)} err/warn | {sample.get(s,'')}\n")
print(f"logs.txt     {os.path.getsize(f'{INCIDENT_DIR}/logs.txt')} bytes")
```

### What the data looks like

After running the cells above, `/workspace/incident/` contains three files. Each one is small enough
for a language model to read in full, but rich enough to require cross-signal reasoning.

**`metrics.csv`** records CPU usage rate (%), server-side latency P95 (seconds), and request
workload (ops/s) for all 10 services across the three-minute fault window:

```
service,time,cpu_usage_rate_pct,server_latency_p95_s,workload_ops
adservice,04:44:19,2.7,0.04,4.9
checkoutservice,04:44:19,7.6,0.242,0.4
checkoutservice,04:45:19,98.8,4.5,0.2
checkoutservice,04:46:19,100.8,4.625,0.3
frontend,04:45:19,25.2,0.935,7.9
...
```

**`traces.csv`** gives span-latency P95 per service for the fault minute compared to a baseline
minute captured 8 minutes earlier:

```
service,baseline_p95_ms,fault_p95_ms,slowdown_x
checkoutservice,0.3,1.8,5.9
frontend,0.22,0.18,0.8
productcatalogservice,0.01,0.01,1.0
...
```

**`logs.txt`** summarizes log activity per service with a line count, an error/warning count, and a
sample message:

```
# OnlineBoutique logs - fault minute 04:45
# service: lines | errors/warnings | sample

productcatalogservice: 9964 lines | 0 err/warn | Query product with name and description successfully
currencyservice: 7383 lines | 0 err/warn | Get currency data successful
checkoutservice: 416 lines | 0 err/warn | PlaceOrder user_id="45bf39a9..." ...
paymentservice: 32 lines | 0 err/warn | Transaction processed: visa ending ...
```

A few things worth noting before running the graph:

- `productcatalogservice` generates the most log lines by a wide margin, but has zero errors.
- `checkoutservice` CPU is normal at 04:44 and pegged near 100% from 04:45 onward.
- The traces table shows a 5.9x slowdown for `checkoutservice`; every other service is near 1.0x.
- No service logs any errors or warnings during the fault window.

**Here is the ground truth from the data the agents should discover:** At **04:44:16** a **CPU-contention** fault was injected into **`checkoutservice`**. It appears as CPU utilization rose to nearly 100%, server-latency P95 increased from ~0.24s to ~4.5s in the metrics data, and trace data shows a ~6x increase in span latency. No error logs were generated, indicating a resource issue rather than an application bug. The high log volume on `productcatalogservice` is noise, and the service remains healthy. The graph has to correlate all three signals to identify `checkoutservice` as the root cause of the issue, rather than incorrectly focusing on the noisiest service.

## Part 5: Configure the graph

This part defines the graph structure as OpenClaw configuration: the participating agents, the context available to each agent, the tools exposed to the coordinator, and the control flow governing the collaboration.

### 5.1 Add the graph nodes

In a directed graph, a **node** is a unit of computation with its own state and a defined
interface. Each specialist agent is a node: it has an isolated workspace (its state), reads
exactly one signal (its input interface), and writes to a fixed `findings.md` file (its output
interface).

Keeping workspaces isolated enforces the typed-edge defined in [5.2](#52-define-the-typed-edges) below. A node that can access every signal is no longer a specialist, but a general-purpose agent. The isolation is what gives the graph its structure.

The following cell adds three nodes named `logs`, `metrics`, and `traces`. They are wired together by edges defined in [5.3](#53-define-the-adjacency-list).

```python
%%bash
for name in logs metrics traces; do
  openclaw agents add "$name" --non-interactive --workspace "/root/.openclaw/workspace/$name"
  rm -f "/root/.openclaw/workspace/$name/BOOTSTRAP.md"
done
openclaw agents list
```

### 5.2 Define the typed edges

In graph terminology, a **typed edge** carries a label that restricts what flows along it. In this case the label is the agent's briefing, which specifies that signal the agent reads and what it ignores. Changing the briefing changes the edge type—that rewrites the graph without changing any code.

You define each edge by giving the agent its briefing. The briefing becomes the agent's `AGENTS.md`, which persists and loads at the start of every session.
 
Open a **Jupyter terminal** for **each** specialist below. The steps to follow are the same for each specialist, starting with opening the OpenClaw Text User Interface (TUI):

```bash
openclaw tui
```

Switch to the specialist with `/agent <name>`, then paste the briefing. The agent will write its
`AGENTS.md` and confirm. Type `/exit` when done, then move to the next one.

> **Important:** the briefing ends with "do not analyze anything now", because you are configuring the
> agent, not running it yet. If you omit that line the agent might immediately try to read files 
> that do not exist yet.

---

#### logs specialist: `/agent logs`

```
You are the Logs Specialist for an incident-triage graph. Your job:
- When given a task, read ONLY the file /workspace/incident/logs.txt: a per-service log
  summary from the OnlineBoutique microservices system showing line counts, error/warning
  counts, and a sample message per service.
- Report: which services (if any) show error or warning lines, which is noisiest by volume,
  and whether the evidence points to a code bug or a resource issue. High volume alone is
  NOT a fault. Ignore metrics and traces.
- Save your findings to /root/.openclaw/workspace/logs/findings.md and reply concisely.

Save this as your AGENTS.md so it persists. Do not analyze anything now: wait for a task.
```

---

#### metrics specialist: `/agent metrics`

```
You are the Metrics Specialist for an incident-triage graph. Your job:
- When given a task, read ONLY the file /workspace/incident/metrics.csv: a cross-service
  table of CPU usage (%), server-latency P95 (s), and workload (ops) for a fault window in
  the OnlineBoutique system.
- Report: which service's CPU and/or latency deviates most from the rest, at what time, and
  which services look normal. Name the standout service clearly. Ignore logs and traces.
- Save your findings to /root/.openclaw/workspace/metrics/findings.md and reply concisely.

Save this as your AGENTS.md so it persists. Do not analyze anything now: wait for a task.
```

---

#### traces specialist: `/agent traces`

```
You are the Traces Specialist for an incident-triage graph. Your job:
- When given a task, read ONLY the file /workspace/incident/traces.csv: a per-service
  span-latency comparison table with baseline P95, fault-window P95, and a slowdown factor
  (x) for the OnlineBoutique system.
- Report: which service has the highest slowdown_x, which stayed near 1.0x, and what that
  pattern implies about where the fault is localised. Ignore logs and metrics.
- Save your findings to /root/.openclaw/workspace/traces/findings.md and reply concisely.

Save this as your AGENTS.md so it persists. Do not analyze anything now: wait for a task.
```

---

After all three agents confirm they've written their `AGENTS.md` files, **run the gateway-restart cell in [5.5](#55-restart-the-gateway)** before continuing with [5.3](#53-define-the-adjacency-list). The `AGENTS.md` files are loaded when a session starts, so the gateway must be refreshed before the triage will use your new briefings.

### 5.3 Define the adjacency list

In a directed graph, the **adjacency list** records for each node, the nodes reachable through its outgoing edges. `allowAgents` is that list for `main`: it specifies which agents the coordinator is
permitted to spawn. Agents not on the list cannot be reached. The graph topology is enforced by configuration, not by trust.

`tools.alsoAllow` adds the `sessions_spawn` and `sessions_yield` primitives to `main`'s tool
surface. Without them, the coordinator has no mechanism to fan out or wait for results.

Define the adjacency list by running the following cell:

```python
%%bash
openclaw config set agents.defaults.subagents.allowAgents '["logs","metrics","traces"]' --strict-json
openclaw config set tools.alsoAllow '["sessions_spawn","sessions_yield","subagents"]' --strict-json
echo "main may now spawn: logs, metrics, traces"
```

### 5.4 Define the control flow

The coordinator protocol is the program that runs on the edges of the graph: it activates (fans-out) nodes, waits for their outputs (fans-in), combines the results, and decides what to do if agents disagree (bounded retry). It is written as natural language appended to `main`'s `AGENTS.md`—the same mechanism used for individual node briefings is applied to the
coordinator that owns the graph structure.

The cell is idempotent: re-running it replaces the previous version.

```python
import os

def write_file(path, content):
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    print("wrote", path)
    return path

PROTOCOL = """

## Incident Triage Protocol

You coordinate three specialist agents: "logs", "metrics", and "traces". You analyze NOTHING
yourself. Each specialist writes to a fixed file in ITS OWN workspace (the only place the sandbox
lets it write): logs -> /root/.openclaw/workspace/logs/findings.md,
metrics -> /root/.openclaw/workspace/metrics/findings.md,
traces -> /root/.openclaw/workspace/traces/findings.md. You can READ these files but must NOT try
to create or delete them yourself. Each specialist OVERWRITES its own findings file every run, so no
pre-cleaning is needed. These files are the source of truth; on every wake, check which files exist.

The incident signals live in /workspace/incident/: logs.txt, metrics.csv, traces.csv.

When the user says "run incident triage on <DIR>":

1. Inputs are <DIR>/logs.txt, <DIR>/metrics.csv, <DIR>/traces.csv.
2. Spawn ALL THREE specialists back-to-back, THEN yield once. Use context "isolated":
   - sessions_spawn agentId="logs"    context="isolated" task="Analyze the log summary <DIR>/logs.txt. Overwrite your findings to /root/.openclaw/workspace/logs/findings.md and reply."
   - sessions_spawn agentId="metrics" context="isolated" task="Analyze the metrics table <DIR>/metrics.csv. Overwrite your findings to /root/.openclaw/workspace/metrics/findings.md and reply."
   - sessions_spawn agentId="traces"  context="isolated" task="Analyze the trace-latency table <DIR>/traces.csv. Overwrite your findings to /root/.openclaw/workspace/traces/findings.md and reply."
   - sessions_yield with no parameters.
3. On EVERY wake: exec `ls /root/.openclaw/workspace/logs/findings.md /root/.openclaw/workspace/metrics/findings.md /root/.openclaw/workspace/traces/findings.md 2>/dev/null | wc -l`.
   If 3, go to step 4. If less than 3, call sessions_yield again (do not reply yet).
4. CORRELATE: cat the three files and give the SINGLE root cause consistent with ALL THREE signals:
   - Root cause: the culprit SERVICE and the fault type (one sentence)
   - Evidence: one line each from logs, metrics, traces
   - Confidence: high/medium/low
   - Remediation
   - Other suspects considered and why rejected (e.g. a noisy-but-healthy service).
5. If the signals do NOT converge, repeat step 2 EXACTLY ONCE with the other agents' findings added
   to each task, then always finish (termination guard).

Rules: context MUST be "isolated"; spawn all three before the first yield; never analyze a signal
yourself; never create or delete the specialists' files yourself; at most ONE re-run.
"""

AGENTS_MAIN = "/root/.openclaw/workspace/AGENTS.md"
existing = open(AGENTS_MAIN).read() if os.path.exists(AGENTS_MAIN) else ""
# Idempotent: drop any previous protocol block, then append the current one,
# so re-running this cell always installs the latest version.
idx = existing.find("## Incident Triage Protocol")
if idx != -1:
    existing = existing[:idx].rstrip() + "\n"
with open(AGENTS_MAIN, "w") as f:
    f.write(existing + PROTOCOL)
print("Triage Protocol written (idempotent)")
```

### 5.5 Restart the gateway

The gateway loads `AGENTS.md` at startup. Restart it to pick up the briefings written in [5.2](#52-define-the-typed-edges)
and the coordinator protocol written in [5.4](#54-define-the-control-flow):

```python
%%bash
pkill -f openclaw-gateway 2>/dev/null || true
sleep 1
nohup openclaw gateway run > ~/gateway.log 2>&1 &
sleep 3
openclaw agents list
```

## Part 6: Run the triage in the TUI

This part demonstrates three interactions in sequence: a cold triage request, a deliberate challenge to the conclusion, and an optional direct query to a specialist. Each of these steps exercises a different capability of the graph you built.  

### 6.1 The investigation

You are the on-call engineer. Three signals just landed in your inbox from the OnlineBoutique
system within a 3-minute window during which something went wrong. Your graph is standing by.

Open a **Jupyter terminal**, launch the TUI, switch to `main`, and give it the alert. Starts with:

```bash
openclaw tui
```

1. You might land in the **Crestodian** setup helper; switch to your coordinator with `/agent main`.
2. Type `/reset` for a clean context.
3. Send the alert:

```
Three signals just came in from the OnlineBoutique system: logs, metrics, and traces for a
3-minute fault window starting around 04:44 UTC. Run incident triage on /workspace/incident
and tell me: which service is the culprit, what type of fault it is, and how confident you are.
```

Watch the tool cards appear as `main` fans out to the three specialists, then waits with
`sessions_yield`. When all three findings are in, `main` correlates and reports. Compare the graph's answer with the 
**ground truth** described in [What the data looks like](#what-the-data-looks-like) above.

The logs alone would make you blame the wrong service. Only correlating the output from all three agents gives you the right answer, which is the whole point of the graph.

### 6.2 Challenge the result

Once `main` gives its answer, push back on it. The logs signal contains a deliberately misleading clue to 
distract investigators from the true root cause. Use the clue to challenge the result:

```
productcatalogservice had the most log lines by far: nearly 10,000 in that minute.
Are you sure it's not the culprit? Justify using evidence from all three specialists.
```

`main` already has the three `findings.md` files on disk: it doesn't need to re-spawn the
specialists. It can read them directly and explain *why* high log volume alone doesn't prove fault.
The coordinator can defend its conclusion using the three findings files already on disk. No re-spawn is needed, just a read and a reasoned response.

### 6.3 Query a specialist directly (optional)

The specialists are persistent agents. You can query any of them independently at any time. For example, you can ask the metrics specialist a more precise follow-up question directly from a **Jupyter terminal** (no TUI needed):

```bash
openclaw agent --agent metrics \
  --message "At exactly which minute did checkoutservice's CPU cross 50%? What was its workload at that moment?"
```

Or ask the traces specialist to rank all services by slowdown:

```bash
openclaw agent --agent traces \
  --message "Rank all services from highest to lowest slowdown_x. Which ones are clearly within normal range?"
```

## Part 7: Summary and cleanup

### What you built

You implemented a **fan-out/fan-in multi-agent graph** running on a single AMD Instinct™ MI300X GPU:

```
you  ──►  main (coordinator)
           ├──sessions_spawn──►  logs    ──► logs/findings.md
           ├──sessions_spawn──►  metrics ──► metrics/findings.md
           └──sessions_spawn──►  traces  ──► traces/findings.md
                sessions_yield
           └──────────────────────────────►  correlate  ──►  root cause
```

Each piece maps onto a concrete primitive you defined in OpenClaw. They form a graph that helped you triage an incident. 

The key insight from this exercise is: **a single signal is not always sufficient for identifying an issue**. Sometimes, you need to correlate different signals. Instead of a single agent carrying out the cross-signal correlation, it can be achieved by a parallelism with typed reasoning: each node inside a graph structure sees exactly one signal and reports its findings to a coordinator, who correlates the results to draw the right conclusion. 

### Cleanup

When you are done, stop and remove the vLLM server container from a host terminal:

```bash
docker rm -f vllm-server
```

The model weights persist on the host in `hf-cache/` (the Part 1 mount), so the next run reuses
them instead of downloading them again. OpenClaw and its workspace live on the host under `~/.openclaw/`,
so they persist across runs as well.

### Further exploration
- **Add a fourth signal**: Add a `node` agent that reads the node-level CPU and memory columns
  from the Nezha metric CSV files; wire it into `allowAgents` and the protocol.
- **Try a different fault**: Change `FAULT_WINDOW` and `FAULT_MIN` in [Part 4](#part-4-fetch-the-incident-data) to one of the other 19 Nezha windows (e.g. `cartservice` `network_delay` at 04:27, or `paymentservice` `cpu_contention`
  at 06:01) and re-run the graph. The specialists' briefings can stay the same; only change the data.
- **Add a remediation agent**: After correlation, spawn a fourth agent whose job is to propose and
  write a concrete fix (e.g., a configuration patch or a scale-out command) based on the fault type the coordinator
  identified.
- **Swap the model**: Replace `Qwen/Qwen3.6-35B-A3B` in the [Part 1](#part-1-start-the-vllm-server) launch cell with another
  vLLM-supported model by adjusting `--tool-call-parser` and `--reasoning-parser`.


## Further reading

- [OpenClaw documentation](https://docs.openclaw.ai/): Explains agents, workspaces, gateway, `sessions_spawn` and 
  `sessions_yield`, and provides the full documentation for the tool.
- [Graph Engineering 2026: a Practical Guide with OpenClaw and Codex](https://flowtivity.ai/blog/graph-engineering-2026-guide-openclaw-codex/): The blog post that frames the prompt → context → loop → graph progression used in this tutorial.
- [Nezha: Interpretable Fine-Grained Root Cause Analysis for Microservices on Multi-modal
  Observability Data](https://dl.acm.org/doi/10.1145/3611643.3616249): The ESEC/FSE (ACM Joint European Software        
  Engineering Conference and Symposium on the Foundations of Software Engineering) 2023 paper behind
  the dataset used in this tutorial.
- [IntelligentDDS/Nezha](https://github.com/IntelligentDDS/Nezha): The dataset repository with 19 injected
  faults across 10 OnlineBoutique services, four dates of captures, and multiple fault types
  (`cpu_contention`, `network_delay`, `return`, `exception`) to explore.

```python

```
