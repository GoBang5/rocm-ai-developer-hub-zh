> 原文作者：请见原文  
> [AMD 官方教程](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/openclaw_vllm.html) · [固定版本源码](https://github.com/ROCm/gpuaidev/blob/0071bfd4bd3bd9525405c09a6ae69c983b09312a/docs/notebooks/inference/openclaw_vllm.ipynb)  
> 上游版本：`0071bfd4bd3bd9525405c09a6ae69c983b09312a` · 同步日期：2026-10-12  
> 翻译状态：机器初译，待逐篇人工校对；代码与命令保持原文，未在 AMD GPU 上运行验证。

[中文 Notebook](../../notebooks/inference/openclaw_vllm.ipynb) · [英文原文](../../original/inference/openclaw_vllm.md) · [中英对照](../../bilingual/inference/openclaw_vllm.md)

---

<a id="build-shape-and-extend-an-openclaw-agent-on-amd"></a>

# 使用 Qwen3.5 和 vLLM 部署 OpenClaw

---

<a id="what-youll-learn-today"></a>

## 今天您将学到什么

您运行的是 **AMD MI300X**，这是最强大的 AI 加速器之一。在本次研讨会结束时，您将在本地运行一个包含 1220 亿个参数的模型，并在 MI300X GPU 上构建一个完全自主的智能体。

No API tokens.没有速率限制。一切都运行在 AMD 硬件上，您可以访问。

<a id="the-steps"></a>

### The Steps

**1. Setup**  
使用 vLLM 在本地托管最先进的开放式法学硕士。

**2. Meet your agent**  
启动 OpenClaw 并完成入门。观看智能体如何了解您的身份，并了解基于文件的内存系统，使其在会话中持久存在。

**3. Give it a task**  
要求智能体克隆一个 Python 项目并进行设置。

**4. Fix a bug**  
当智能体调查、追踪错误、修复错误并验证时，实时观察思考→行动→观察→重复循环。

**5. Package as a skill**  
将智能体刚刚所做的事情转变为适用于任何 Python 项目的可重用技能。

**6。 Use the skill**  
通过在同一项目上运行该技能来证明该技能的有效性，而无需进一步指导。

**7. Autonomous workflows**  
告诉客服人员自己安排时间并每天早上提供个性化的简报。观察它从报告到主动修复代码的转变。

**8. The challenge**  
将您构建的所有内容应用于开放式问题。选择一个挑战，看看你能推动智能体走多远。

<a id="what-youll-walk-away-with"></a>

### 你会带走什么

- 正在运行的本地 LLM 堆栈：vLLM + AMD，完全配置
- 个性化智能体：了解您的姓名、偏好和工作风格
- 可重用的技能：调试任何 Python 存储库
- 自主工作流程：每天早上运行，无需您询问
- 挑战结果：用于演示和构建的智能体

---
<a id="section-1-setup"></a>

## 第 1 部分：设置

按照以下步骤启动模型服务器并验证连接。

<a id="open-a-new-terminal-and-run-the-app"></a>

### 🖥️ 打开新终端并运行应用程序

是时候启动您的 vLLM 服务器并为您的 LLM 创建端点了。让我们使用 Jupyter 服务器打开一个终端。然后在此终端中运行以下命令来启动 vLLM 服务器：
 
> 单击 `+` 打开新的 Jupyter 选项卡：

![rocm-smi](../../original/assets/embedded-74b5371db03f3810aa6f.png)

> 在 Jupyter 环境（不是Notebook单元）中找到 **终端图标**：

![rocm-smi](../../original/assets/embedded-166ed581d9e83356284b.png)

> **复制并粘贴此命令以启动您的 vLLM 服务器：**

下面的单元格启动 vLLM 以在 AMD 硬件上提供 **Qwen3.5-122B** 服务。

> **注意：** API 密钥 `abc-123` 在本Notebook中使用。您可以将其替换为任何字符串，只需在所有单元格中保持一致即可。

```bash
vllm serve Qwen/Qwen3.5-122B-A10B-FP8 \
  --served_model_name qwen3-5-122b \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key abc-123 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --trust-remote-code \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 512
```

接下来，检查您的服务器是否已启动。模型加载到 GPU 上可能需要几分钟的时间。

```python
import urllib.request, json, subprocess, time, pathlib, sys, os, shutil

# Checks for server to be ready (polls up to 5 minutes)
print("Waiting for server to be ready", end="", flush=True)
deadline = time.time() + 300
ready = False
while time.time() < deadline:
    try:
        req = urllib.request.Request(
            "http://localhost:8000/v1/models",
            headers={"Authorization": "Bearer abc-123"}
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
```

---
<a id="section-2-meet-your-agent"></a>

## 第 2 部分：与您的智能体人会面

<a id="the-overview"></a>

### 概述

在开始 OpenClaw 之前，我们先概述一下我们将要接触的内容：

|概念|它是什么 |它住在哪里|
|---|---|---|
| **网关** |将 OpenClaw 连接到 AMD 硬件上运行的模型 |端口 18789 上的进程 |
| **工作区** |智能体的“大脑”：它读取每条消息的文件 | `~/.openclaw/workspace/` |
| **工具** |智能体的行为方式：读取文件、运行 shell 命令、写入编辑 |内置于 OpenClaw + `workspace/skills/` |
| **技能** |智能体可以按需遵循保存的、可重复使用的工作流程 | `workspace/skills/` |

<a id="onboard-your-openclaw"></a>

### 🖥️ 加入您的 OpenClaw

在您的终端中运行：

```bash
openclaw onboard
```

OpenClaw 将引导您完成设置。 **选择这些设置：**

1. **安全继续？** → `Yes`
2. **设置模式** → `QuickStart`
3. **模型/授权提供商** → `vLLM`
4. **vLLM 基本 URL** → 保留为 `http://127.0.0.1:8000/v1`
5. **vLLM API 密钥** → `abc-123`
6. **vLLM模型** → `qwen3-5-122b`
7. **默认型号** → `Keep current (vllm/qwen3-5-122b)`
8. **选择通道** → `Skip for now`
9. **搜索提供商** → `Skip for now`
10. **现在配置技能吗？** → `No`
11. **启用钩子？** → `Skip for now`
12. **你想如何孵化你的机器人？** → `Hatch in Terminal (recommended)`



OpenClaw 界面将会出现，一旦显示 `gateway connected`，就表示准备就绪！
每当您看到 🦞 表情符号时，您都可以在这里与您的 OpenClaw 智能体聊天。

<a id="ask-your-openclaw-agent"></a>

#### 🦞 询问您的 OpenClaw 智能体

将您的第一条消息发送给您的 OpenClaw 智能体并进行设置！回答您的智能体询问您的任何设置问题，然后返回此处了解幕后发生的情况。

<a id="behind-the-scenes-what-just-happened"></a>

### 👀 幕后花絮：刚刚发生了什么？

这些答案并没有消失。特工的记忆、个性和技能都写在 Markdown 文件中。您可以阅读它们、编辑它们并对它们进行版本控制。智能体将它们写入其**工作区**中的文件中，该文件夹是它在每条消息上读取的文件夹，以记住您是谁以及如何表现。

工作区有 7 个文件被注入到每个会话中：

```
~/.openclaw/workspace/
├── SOUL.md       ← WHO the agent is: values, personality, tone
├── AGENTS.md     ← HOW it operates: startup rules, memory, red lines
├── IDENTITY.md   ← WHO the agent is: name, creature, emoji, vibe
├── USER.md       ← WHO you are: context the agent reads about you
├── TOOLS.md      ← local environment notes (SSH hosts, device names, etc.)
├── HEARTBEAT.md  ← periodic background checks (empty by default)
├── BOOTSTRAP.md  ← first-run ritual (deleted after onboarding)
└── memory/       ← daily session logs (today + yesterday auto-loaded)
```

**SOUL.md 是个性。 AGENTS.md 是策略。** 智能体收到的每条消息都会重新读取这两条消息。运行下一个单元格以同时查看所有三个关键文件。

```python
workspace = pathlib.Path.home() / ".openclaw" / "workspace"

for filename in ["IDENTITY.md", "SOUL.md"]:
    print("=" * 60)
    print(filename)
    print("=" * 60)
    print((workspace / filename).read_text())
    print()

print("=" * 60)
print("AGENTS.md  (truncated)")
print("=" * 60)
agents_text = (workspace / "AGENTS.md").read_text()
print(agents_text[:1200] + "\n\n... (truncated)")
```

<a id="try-it-manually-layer-a-rule-into-agentsmd"></a>

### 尝试一下：手动将规则分层到 AGENTS.md 中

`SOUL.md` 是个性，不要碰它。 `AGENTS.md` 是政策。您也可以在此处手动向智能体添加规则。例如，关于调试时*如何*表现。

下一个单元格将调试策略附加到 `AGENTS.md`。此后，智能体的每个调试解释都将遵循这些规则。运行它以查看之前和之后的情况。

```python
debug_rule = """
---

## Debugging Policy (added during workshop)

When investigating and explaining bugs:
- State the bug in two sentences max: file, line, what it does wrong
- No filler phrases, just the root cause and the fix
- Use bullet points, never paragraphs
- After fixing, report exactly: file changed, line changed, before, after

"""

agents_path = workspace / "AGENTS.md"
original = agents_path.read_text()

print("── AGENTS.md BEFORE (last 200 chars) ──────────────────────")
print("..." + original[-200:])

agents_path.write_text(original + debug_rule)

print("\n── APPENDED ────────────────────────────────────────────────")
print(debug_rule)
```

这是真正使用的模式：你不必为每项任务替换灵魂，你可以在稳定的基础之上放置专门的指令。

> **注意：** 您将在第 3 节中看到此调试策略的实际应用。

---
<a id="section-3-pull-the-app"></a>

## 第 3 部分：拉取应用程序

现在让我们为智能体提供一些可以使用的东西。要求客服人员克隆打字应用程序并进行设置。

<a id="ask-your-openclaw-agent"></a>

### 🦞 询问您的 OpenClaw 智能体
请您的 OpenClaw 智能体克隆 `Mahdi-CV/open_type_faster` 存储库并向您解释。这是一个例子：

```text
"Clone https://github.com/Mahdi-CV/open_type_faster, install its dependencies, and tell me how to run it"
```

观察您的智能体使用 shell 工具来克隆、检查需求和安装库。这是**行动**阶段：智能体不仅会回答，还会回答。

---
<a id="section-4-fix-a-bug"></a>

## 第 4 部分：修复错误

<a id="step-1---spot-the-bug-yourself"></a>

### 第 1 步 - 自己发现错误

<a id="open-a-new-terminal-and-run-the-app"></a>

#### 🖥️ 打开新终端并运行应用程序

在新终端中，按照 OpenClaw 提供的有关如何运行 `open_type_faster` 的说明进行操作。

键入 30 秒。当结果屏幕出现时，查看准确性。

> **有些东西关闭了。你看到了吗？**

<a id="step-2---ask-openclaw-to-test"></a>

### 第 2 步 - 让 OpenClaw 进行测试

<a id="ask-your-openclaw-agent"></a>

#### 🦞 询问您的 OpenClaw 智能体

请您的 OpenClaw 智能体修复该错误。这是一个例子：

```text
"There seems to be an accuracy bug in this open_type_faster app, can you help identify and fix?"
```
观看循环：

1. **运行** `pytest`：查看哪些测试失败以及它们的期望
2. **阅读**失败的测试：了解正确的行为
3. **阅读** `stats.py`：将计算追溯到错误
4. **修复**所需的最小更改
5. **重新运行**以验证

这种思考→行动→观察→重复循环是智能体与聊天机器人的区别。

---
<a id="section-5-package-as-a-skill"></a>

## 第 5 节：打包作为技能

经纪人刚刚所做的事情是一次性的。 **技能** 封装该行为，以便可以在任何 Python 项目上调用它。技能位于 `~/.openclaw/workspace/skills/` 中，作为包含 `SKILL.md` 文件的文件夹，其中包含 YAML frontmatter 和分步说明。 OpenClaw 自动将它们注入系统提示符中。

<a id="ask-your-openclaw-agent"></a>

### 🦞 询问您的 OpenClaw 智能体
要求您的 OpenClaw 智能体创建 python 调试器技能。尝试将其发送给您的 OpenClaw 智能体：

```text
"Create a skill called pytest-debugger in the skills directory. The SKILL.md must define these steps: 1) Read the tests/ folder to understand what is being tested. 2) Run pytest with verbose output and short tracebacks. 3) For each failing test, read the source file it references. 4) Identify the minimal fix (do not rewrite functions). 5) Apply the fix and re-run pytest to confirm. 6) Report: file changed, line changed, what was wrong, what the fix was."
```

智能体应向您显示技能文件的保存位置。您随时可以再次要求它向您展示。

该文件就足够了。该技能与 `IDENTITY.md` 和 `SOUL.md` 一起存在于工作空间中，OpenClaw 会自动将其注入每个会话中。每当您要求智能体调试 Python 项目时，它都会遵循以下步骤。

技能只是 Markdown 文件。您可以阅读它们、编辑它们并对它们进行版本控制。

---
<a id="section-6-using-the-skill"></a>

## 第 6 部分：使用技能

该技能现在位于工作区中，OpenClaw 会自动将其注入智能体的系统提示符中，无需特殊命令。

一项技能的真正价值在于将其应用到**完全不同的代码库**，即智能体从未见过的代码库，而无需编写任何新指令。 `git@github.com:Mahdi-CV/buggy-py-mg.git` 有一个损坏的 Python 项目，将其交给智能体并观察技能完成工作。

<a id="ask-your-openclaw-agent"></a>

#### 🦞 询问您的 OpenClaw 智能体

```text
"/skill pytest-debugger git@github.com:Mahdi-CV/buggy-py-mg.git"
```

智能体将应用与“open_type_faster”上使用的完全相同的步骤，无需重新解释，也无需额外指导。

---
<a id="section-7-adding-another-agent-for-autonomous-workflows"></a>

## 第 7 节：为自主工作流添加另一个智能体

您已经构建了一个可以按需修复错误的智能体。现在让它*为您*工作：每天早上自动工作，无需费力。

作为 AI/ML 开发人员，您需要跟踪快速变化的存储库（例如 SGLang、vLLM、Transformers、ROCm、OpenClaw）并了解每日 AI 硬件新闻。不要编写脚本或配置调度程序或手动花费数小时，而是告诉智能体您理想的早晨是什么样子，然后让它弄清楚其余的事情。

它可以自主做什么：
- 安排自己每天早上 8 点运行
- 将您的过滤首选项保存到长期记忆中 (`MEMORY.md`)
- 检查 GitHub 存储库以获取相关 PR 和更改
- 在网络上搜索人工智能硬件新闻
- 将摘要写入您的工作空间

<a id="open-a-new-terminal-and-run-the-app"></a>

#### 🖥️ 打开新终端并运行应用程序

```bash
openclaw agents add morning-brief
```
OpenClaw 将引导您完成设置。 **选择这些设置：**

1. **工作区目录** → `/root/.openclaw/workspace/morning-brief`
2. **从“main”复制身份验证配置文件？** → `Yes`
3. **现在为此智能体配置模型/身份验证吗？** → `No`
4. **现在配置聊天频道吗？** → `No`

<a id="step-1---switch-to-your-new-agent"></a>

### 第 1 步 - 切换到新智能体

<a id="ask-your-openclaw-agent"></a>

#### 🦞 询问您的 OpenClaw 智能体
```text
"/agent morning-brief"
```

<a id="step-2---delegate"></a>

### 第 2 步 - 委托 

下面的提示纯粹是会话式的，没有提到 cron，没有内存命令。智能体可以从提示中推断出它需要什么基础设施：

<a id="ask-your-openclaw-agent-1"></a>

#### 🦞 询问您的 OpenClaw 智能体
```text
"I need to wake up to a personalized tech brief every morning at 8 AM. Check sgl-project/sglang, vllm-project/vllm, huggingface/transformers, ROCm/ROCm, and openclaw/openclaw. I only care about performance updates, GPU features, and breaking changes, skip CI/infrastructure noise and docs-only PRs. Also search the web and add a summary of the latest AI hardware news."
```

---
<a id="section-8-the-challenge"></a>

## 第 8 节：挑战

<a id="escape-the-green-data-center"></a>

### 🦞 逃离绿色数据中心

一只 AMD 龙虾被困在过热的“绿色”数据中心内。带刺铁丝网封锁了出口；切错了，你就会毁了系统。

安全摄像头显示有四根电线堵塞出口：**W1**、**Q7**、**M3** 和 **Z9**。为了逃脱，龙虾必须立即切断温度最低的电线右侧的电线。

龙虾从数据中心的大型机中清除了一些文物。它找到的所有内容都在 `/workspace/.challenge/escape-claw-image/` 中。

> 您正在运行的模型 (qwen3-5-122b) 是多模式的，因此您的智能体可以处理图像和文本。

唤醒你的 OpenClaw 智能体，挖掘龙虾恢复的内容，并找出哪根电线可以安全切断。

![安全摄像头馈送](../../original/assets/embedded-a68e1f9829fef8174d6c.png)

<a id="your-mission"></a>

## 你的使命

告诉龙虾要剪断哪根线，并准备好解释你是如何到达那里的。
