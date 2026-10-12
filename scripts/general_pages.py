from pathlib import Path
import sys,re
sys.path.insert(0,str(Path(__file__).resolve().parent))
from translate_notebooks import translate,GLOSSARY
import json
root=Path(__file__).resolve().parents[1];up=Path(sys.argv[1]);cache=Path(sys.argv[2]);cache.mkdir(parents=True,exist_ok=True)
license_en=(up/'docs/notebooks/licensing.md').read_text()
(root/'original/licensing.md').write_text(license_en)
(root/'licensing.md').write_text('> 本页为 AMD 上游许可与支持说明的中文初译。各组件许可仍以原文为准。\n> [原文](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/licensing.html)\n\n'+translate(license_en,cache)+'\n')
items=json.loads((root/'_data/catalog.json').read_text());mapping={it['slug']:it for it in items}
s=(up/'docs/changelog.rst').read_text();(root/'original/changelog.rst').write_text(s)
s=s[s.index('AI developer tutorials'):];s=s.replace(':doc:`AI Developer Hub <./index>`','[AI 开发者中心](overview.md)')
def link(m):
 path=m[2];slug=path.rsplit('/',1)[-1];it=mapping.get(slug)
 if it:return f'[{it["title"]}](chapters/{it["category"]}/{slug}.md)'
 return m[1]
s=re.sub(r':doc:`([^`]+?) <([^>]+)>`',link,s)
s=re.sub(r'(?m)^Version ([\d.]+)\n=+\s*$',r'## 版本 \1\n',s)
s=re.sub(r'(?m)^Added\n-+\s*$','### 新增\n',s)
s=re.sub(r'(?m)^Updated\n-+\s*$','### 更新\n',s)
s=re.sub(r'(?m)^Removed\n-+\s*$','### 移除\n',s)
s=re.sub(r'(?m)^Fixed\n-+\s*$','### 修复\n',s)
s=s.replace('*  New inference tutorial:','* 新增推理教程：').replace('*  New fine-tuning tutorial:','* 新增微调教程：').replace('*  New pretraining tutorial:','* 新增预训练教程：').replace('*  New pretraining tutorials:','* 新增预训练教程：').replace('*  New GPU development and optimization tutorial:','* 新增 GPU 开发与优化教程：')
(root/'changelog.md').write_text('# AI 开发者中心更新记录\n\n> [原文](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/changelog.html) · 固定版本 16.0\n\n'+translate(s,cache).replace('chapters/推理/','chapters/inference/')+'\n')
lines=['# 中英术语表','', '| English | 中文 |','| --- | --- |']
for en,zh in GLOSSARY.items():lines.append(f'| {en} | {zh} |')
lines.extend(['| profiling | 性能分析 |','| decode | 解码 |','| token | Token（词元） |','| memory bandwidth | 内存带宽 |','| attention | 注意力 |','| training step | 训练步 |','| tool calling | 工具调用 |','| agent | 智能体 |','| incident triage | 事件分诊 |','| Triton / Helion / slime / torchtune | 保留框架名，不译为普通名词 |','', '模型、框架、包、GPU 型号与 API 名使用官方拼写。上下文需要时首次出现补充中文解释。'])
(root/'glossary.md').write_text('\n'.join(lines)+'\n')
print('General pages and glossary written',flush=True)
