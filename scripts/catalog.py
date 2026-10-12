from pathlib import Path
import re,json
TITLES = [
'ChatQnA：vLLM 部署与性能评估','使用 ComfyUI 生成视频','在 AMD Instinct GPU 上通过 Python 运行 ComfyUI 生成工作流','在 CPU 或 GPU 上运行 DeepSeek Janus Pro','使用 vLLM V1 运行 DeepSeek-R1','使用 vLLM、PydanticAI 与 MCP 构建 AI 智能体','使用 Hugging Face Transformers 进行推理','使用 vLLM 部署模型','使用 vLLM：从聊天机器人到说唱机器人','使用 LlamaIndex 和 Ollama 构建 RAG','使用 vLLM 与视觉语言模型进行 OCR','为语音助手构建 AI 流水线','使用 vLLM 进行投机解码','使用 SGLang 运行 DeepSeek-R1','使用 SGLang 分离预填充与解码','使用 SGLang 多 Token 预测加速 DeepSeek-V3 推理','使用 Google ADK 与 A2A 协议构建多智能体','使用 Qwen3.5 和 vLLM 部署 OpenClaw','使用 OpenClaw 构建多智能体事件分诊系统','在 AMD Instinct GPU 上分析并优化 AI 智能体性能',
'使用 DiffSynth-Studio 定制 Qwen-Image','使用 PEFT 微调视觉语言模型','使用 torchtune 微调 Llama-3.1 8B','使用 Llama-Factory 微调 Llama-3.1 8B','使用 Unsloth 进行 GRPO 训练','使用 slime 进行 GRPO 训练',
'Megatron-LM 训练环境配置','使用 Megatron-LM 训练大语言模型','使用 torchtitan 训练 Llama-3.1 8B','使用 PyTorch 训练自定义扩散模型','使用 SpecForge 训练投机解码草稿模型','使用 TorchTitan 预训练模型','使用 Primus 训练模型','SE(3)-Transformer 概览','使用 Triton Inference Server 训练与部署 FinalNet CTR 模型',
'使用 Quark MXFP4 量化运行 vLLM','使用 Helion 开发与评估 GPU 内核','AITER 库的 MLA 解码内核','使用 Triton 开发与优化内核','使用 vLLM 分析 Llama-4 推理性能','使用 AMD Quark 为 vLLM 进行 FP8 量化','在 AMD CDNA4 GPU 上优化 FP8 GEMM']
CATEGORIES={'inference':'推理','fine_tune':'微调','pretrain':'预训练','gpu_dev_optimize':'GPU 开发与优化'}

def create(upstream,root):
 toc=(upstream/'docs/sphinx/_toc.yml.in').read_text()
 entries=re.findall(r'- file: notebooks/([^\n]+\.ipynb)\n\s+title: ([^\n]+)',toc)
 assert len(entries)==len(TITLES)==42
 sha='0071bfd4bd3bd9525405c09a6ae69c983b09312a'
 items=[]
 for i,((rel,en),zh) in enumerate(zip(entries,TITLES),1):
  p=Path(rel); d=json.loads((upstream/'docs/notebooks'/p).read_text());text='\n'.join(''.join(c['source']) for c in d['cells'] if c['cell_type']=='markdown')
  author=re.search(r'(?mi)\*\*Authors?\*\*\s*:\s*([^\n]+)',text)
  item={'number':i,'category':p.parent.name,'category_zh':CATEGORIES[p.parent.name],'slug':p.stem,'title':zh,'title_en':en,'source_path':'docs/notebooks/'+rel,'upstream_sha':sha,'source_url':f'https://github.com/ROCm/gpuaidev/blob/{sha}/docs/notebooks/{rel}','official_url':'https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/'+rel.replace('.ipynb','.html'),'author':author[1].strip() if author else '请见原文','status':'machine_draft','reviewed':False,'runtime_verified':False}
  items.append(item)
 (root/'_data/catalog.json').write_text(json.dumps(items,ensure_ascii=False,indent=2)+'\n')
 return items
if __name__=='__main__':
 import sys
 create(Path(sys.argv[1]),Path(__file__).resolve().parents[1])
