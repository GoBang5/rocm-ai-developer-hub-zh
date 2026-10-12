# 目录

- [项目简介](README.md)
- [教程筛选](tutorial-selector.md)
- [术语表](glossary.md)

## 推理

1. [ChatQnA：vLLM 部署与性能评估](chapters/inference/opea_deployment_and_evaluation.md)
2. [使用 ComfyUI 生成视频](chapters/inference/t2v_comfyui_radeon.md)
3. [在 AMD Instinct GPU 上通过 Python 运行 ComfyUI 生成工作流](chapters/inference/t2v_comfyui_api_mode_instinct.md)
4. [在 CPU 或 GPU 上运行 DeepSeek Janus Pro](chapters/inference/deepseek_janus_cpu_gpu.md)
5. [使用 vLLM V1 运行 DeepSeek-R1](chapters/inference/vllm_v1_DSR1.md)
6. [使用 vLLM、PydanticAI 与 MCP 构建 AI 智能体](chapters/inference/build_airbnb_agent_mcp.md)
7. [使用 Hugging Face Transformers 进行推理](chapters/inference/1_inference_ver3_HF_transformers.md)
8. [使用 vLLM 部署模型](chapters/inference/3_inference_ver3_HF_vllm.md)
9. [使用 vLLM：从聊天机器人到说唱机器人](chapters/inference/rapbot_vllm.md)
10. [使用 LlamaIndex 和 Ollama 构建 RAG](chapters/inference/rag_ollama_llamaindex.md)
11. [使用 vLLM 与视觉语言模型进行 OCR](chapters/inference/ocr_vllm.md)
12. [为语音助手构建 AI 流水线](chapters/inference/voice_pipeline_rag_ollama.md)
13. [使用 vLLM 进行投机解码](chapters/inference/speculative_decoding_deep_dive.md)
14. [使用 SGLang 运行 DeepSeek-R1](chapters/inference/deepseekr1_sglang.md)
15. [使用 SGLang 分离预填充与解码](chapters/inference/SGlang_PD_Disagg_On_AMD_GPU.md)
16. [使用 SGLang 多 Token 预测加速 DeepSeek-V3 推理](chapters/inference/mtp.md)
17. [使用 Google ADK 与 A2A 协议构建多智能体](chapters/inference/power-Google-ADK-on-AMD-platform-and-local-LLMs.md)
18. [使用 Qwen3.5 和 vLLM 部署 OpenClaw](chapters/inference/openclaw_vllm.md)
19. [使用 OpenClaw 构建多智能体事件分诊系统](chapters/inference/graph_multiagent_openclaw.md)
20. [在 AMD Instinct GPU 上分析并优化 AI 智能体性能](chapters/inference/agentic_ai_profiling.md)

## 微调

21. [使用 DiffSynth-Studio 定制 Qwen-Image](chapters/fine_tune/qwen_image.md)
22. [使用 PEFT 微调视觉语言模型](chapters/fine_tune/fine_tuning_lora_qwen2vl.md)
23. [使用 torchtune 微调 Llama-3.1 8B](chapters/fine_tune/torchtune_llama3.md)
24. [使用 Llama-Factory 微调 Llama-3.1 8B](chapters/fine_tune/llama_factory_llama3.md)
25. [使用 Unsloth 进行 GRPO 训练](chapters/fine_tune/unsloth_Llama3_1_8B_GRPO.md)
26. [使用 slime 进行 GRPO 训练](chapters/fine_tune/slime_qwen3_4B_GRPO.md)

## 预训练

27. [Megatron-LM 训练环境配置](chapters/pretrain/setup_tutorial.md)
28. [使用 Megatron-LM 训练大语言模型](chapters/pretrain/train_llama_mock_data.md)
29. [使用 torchtitan 训练 Llama-3.1 8B](chapters/pretrain/torchtitan_llama3.md)
30. [使用 PyTorch 训练自定义扩散模型](chapters/pretrain/ddim_pretrain.md)
31. [使用 SpecForge 训练投机解码草稿模型](chapters/pretrain/SpecForge_SGlang.md)
32. [使用 TorchTitan 预训练模型](chapters/pretrain/torchtitan_deepseek.md)
33. [使用 Primus 训练模型](chapters/pretrain/training_with_primus.md)
34. [SE(3)-Transformer 概览](chapters/pretrain/se3transform_intro.md)
35. [使用 Triton Inference Server 训练与部署 FinalNet CTR 模型](chapters/pretrain/triton_inference_server_benchmark.md)

## GPU 开发与优化

36. [使用 Quark MXFP4 量化运行 vLLM](chapters/gpu_dev_optimize/mxfp4_quantization_quark_vllm.md)
37. [使用 Helion 开发与评估 GPU 内核](chapters/gpu_dev_optimize/helion_gpu_kernel_dev.md)
38. [AITER 库的 MLA 解码内核](chapters/gpu_dev_optimize/aiter_mla_decode_kernel.md)
39. [使用 Triton 开发与优化内核](chapters/gpu_dev_optimize/triton_kernel_dev.md)
40. [使用 vLLM 分析 Llama-4 推理性能](chapters/gpu_dev_optimize/llama4_profiling_vllm.md)
41. [使用 AMD Quark 为 vLLM 进行 FP8 量化](chapters/gpu_dev_optimize/fp8_quantization_quark_vllm.md)
42. [在 AMD CDNA4 GPU 上优化 FP8 GEMM](chapters/gpu_dev_optimize/fp8_gemm_hip_cdna4.md)

