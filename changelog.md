# AI 开发者中心更新记录

> [原文](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/changelog.html) · 固定版本 16.0

AI开发者教程可在[AI开发者中心](overview.md)获取。

## 版本 16.0

###新增

* 新增推理教程：

   * [在 AMD Instinct GPU 上分析并优化 AI 智能体性能](chapters/inference/agentic_ai_profiling.md)

## 版本 15.0

###新增

* 新增推理教程：

   * [使用OpenClaw构建多智能体事件分诊系统](chapters/inference/graph_multiagent_openclaw.md)

## 版本 14.0

###新增

* 新增预训练教程：

   * [使用 Triton 推理 服务器训练与部署 FinalNet CTR 模型](chapters/pretrain/triton_inference_server_benchmark.md)

* 新增推理教程：

   * [在AMD Instinct GPU上通过Python运行ComfyUI生成工作流](chapters/inference/t2v_comfyui_api_mode_instinct.md)

## 版本 13.0

###新增

* 新增推理教程：

   * [使用Qwen3.5和vLLM部署OpenClaw](chapters/inference/openclaw_vllm.md)

* 新增GPU开发与优化教程：

   * [在 AMD CDNA4 GPU 上优化 FP8 GEMM](chapters/gpu_dev_optimize/fp8_gemm_hip_cdna4.md)

## 版本 12.0

###新增

* 新教程在登陆页面上突出显示。
* 教程库存选择器。
* 知识水平 和各个教程的作者信息。

## 版本 11.0

###新增

* 新增预训练教程：

   * [SE(3)-Transformer 概述](chapters/pretrain/se3transform_intro.md)

## 版本 10.0

###新增

* 新增教程：

   * [使用史莱姆进行GRPO训练](chapters/fine_tune/slime_qwen3_4B_GRPO.md)

## 版本 9.0

###新增

* 新增教程：

   * [使用 DiffSynth-Studio 定制 Qwen-Image](chapters/fine_tune/qwen_image.md)

* 新增GPU开发与优化教程：

   * [使用 Helion 开发与评估 GPU 内核](chapters/gpu_dev_optimize/helion_gpu_kernel_dev.md)

## 版本 8.0

###新增

* 新增预训练教程：

   * [使用 TorchTitan 预训练模型](chapters/pretrain/torchtitan_deepseek.md)
   * [使用 Primus 训练模型](chapters/pretrain/training_with_primus.md)

## 版本 7.0

###新增

* 新增预训练教程：

   * [使用 SpecForge 训练投机解码草稿模型](chapters/pretrain/SpecForge_SGlang.md)

* 新增GPU开发与优化教程：

   * [使用 Quark MXFP4 量化运行 vLLM](chapters/gpu_dev_optimize/mxfp4_quantization_quark_vllm.md)

## 版本 6.0

###新增

* 新推理教程：

   * [使用 SGLang 多代币预测加速 DeepSeek-V3 推理](chapters/inference/mtp.md)
   * [使用 Google ADK 与 A2A 协议构建多智能体](chapters/inference/power-Google-ADK-on-AMD-platform-and-local-LLMs.md)

### 移除

* 以下 推理 教程已被删除：

   * 抱脸TGI
   * 骆驼堆栈

* 以下 微调 教程已被删除：

   * LoRA 法学硕士
   * QLoRA 法学硕士 

* 以下 预训练 教程已删除：

   * 带有 PyTorch FSDP 的 OLMo 模型

## 版本 5.1

###新增

* 新推理教程：

   * [使用SGLang分离预填充与解码](chapters/inference/SGlang_PD_Disagg_On_AMD_GPU.md)

改变了
--------

* 更新了[使用 vLLM、PydanticAI 与 MCP 构建 AI 智能体](chapters/inference/build_airbnb_agent_mcp.md) 教程中的 ``pydantic_ai`` 参数和 Node 版本。

## 版本 5.0

###新增

* 新推理教程：

   * [ChatQnA：vLLM部署与性能评估](chapters/inference/opea_deployment_and_evaluation.md)
   * [使用 ComfyUI 生成视频](chapters/inference/t2v_comfyui_radeon.md)
   * [在CPU或GPU上运行DeepSeek Janus Pro](chapters/inference/deepseek_janus_cpu_gpu.md)
   * [使用 vLLM V1 运行 DeepSeek-R1](chapters/inference/vllm_v1_DSR1.md)

* 新增GPU开发与优化教程：

   * [AITER库的MLA解码内核](chapters/gpu_dev_optimize/aiter_mla_decode_kernel.md)

## 版本 4.0

###新增

* 新增推理教程：

   * [使用 vLLM、PydanticAI 与 MCP 构建 AI 智能体](chapters/inference/build_airbnb_agent_mcp.md)
* 新的GPU开发和优化教程：

   * [使用 Triton 开发与优化内核](chapters/gpu_dev_optimize/triton_kernel_dev.md)
   * [使用 vLLM 分析 Llama-4 推理性能](chapters/gpu_dev_optimize/llama4_profiling_vllm.md)
   * [使用 AMD Quark 为 vLLM 进行 FP8 量化](chapters/gpu_dev_optimize/fp8_quantization_quark_vllm.md)

改变了
--------

* DDIM 预训练 教程重命名为 [使用 PyTorch 训练自定义扩散模型](chapters/pretrain/ddim_pretrain.md)
   有一些细微的变化。
  
##版本3.1

###新增

* 新增教程：

   * [使用 Unsloth 进行 GRPO 训练](chapters/fine_tune/unsloth_Llama3_1_8B_GRPO.md)

## 版本 3.0

###新增

* 新推理教程：

   * [使用 vLLM 进行投机解码](chapters/inference/speculative_decoding_deep_dive.md)
   * 骆驼栈
   * [使用SGLang运行DeepSeek-R1](chapters/inference/deepseekr1_sglang.md)


* 新增教程：
  
   * [使用 Llama-Factory 参数 Llama-3.1 8B](chapters/fine_tune/llama_factory_llama3.md)

* 新增预训练教程：

   * [使用 PyTorch 训练认知模型](chapters/pretrain/ddim_pretrain.md)

## 版本 2.0

###新增

* 新的推理教程：

   * [使用vLLM与视觉语言模型进行OCR](chapters/inference/ocr_vllm.md)
   * [为语音助手构建 AI 模拟](chapters/inference/voice_pipeline_rag_ollama.md)

* 新增教程：
  
   * [使用 torchtune 参数 Llama-3.1 8B](chapters/fine_tune/torchtune_llama3.md)

* 新增预训练教程：

   * [使用 torchtitan 训练 Llama-3.1 8B](chapters/pretrain/torchtitan_llama3.md)

改变了
--------

* 更新了一些指南支持的 ROCm 版本
* 额外的小改动

## 版本 1.0

###新增

* 新推理教程：

   * [使用抱脸变形金刚进行推理](chapters/inference/1_inference_ver3_HF_transformers.md)
   * 抱脸TGI
   * [使用 vLLM 部署模型](chapters/inference/3_inference_ver3_HF_vllm.md)
   * [使用 vLLM：从聊天机器人到说唱机器人](chapters/inference/rapbot_vllm.md)
   * [使用 LlamaIndex 和 Ollama 构建 RAG](chapters/inference/rag_ollama_llamaindex.md)

* 新的微调教程：
  
   * [使用 PEFT 语言视觉模型](chapters/fine_tune/fine_tuning_lora_qwen2vl.md)
   * LoRA 法学硕士
   * QLoRA 法学硕士

* 新增预训练教程：

   * 带有 PyTorch FSDP 的 OLMo 模型
   * [Megatron-LM 训练环境配置](chapters/pretrain/setup_tutorial.md)
   * [使用Megatron-LM 训练大语言模型](chapters/pretrain/train_llama_mock_data.md)

