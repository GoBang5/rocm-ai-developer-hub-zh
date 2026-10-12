#!/usr/bin/env python3
from pathlib import Path
import json,re,hashlib,sys
ROOT=Path(__file__).resolve().parents[1]
UP=Path(sys.argv[1]);items=json.loads((ROOT/'_data/catalog.json').read_text())
selector=(UP/'docs/tutorial-selector.rst').read_text()
for it in items:
 nb=json.loads((UP/it['source_path']).read_text());text='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='markdown')
 level=re.search(r'(?i)\*\*Knowledge level\*\*\s*:\s*(\w+)',text)
 it['level']=(level[1].lower() if level else 'unknown')
 # Selector is the authoritative fallback for notebooks without a level annotation.
 if it['level']=='unknown':
  for m in re.finditer(r'\.\. selected:: level=(beginner|intermediate|advanced)[^\n]*\n(.*?)(?=\n\.\. selected::|\Z)',selector,re.S):
   if f'/{it["slug"]}>' in m[2]:it['level']=m[1];break
 it['original_sha256']=hashlib.sha256((UP/it['source_path']).read_bytes()).hexdigest()
 it['cell_count']=len(nb['cells']);it['markdown_cell_count']=sum(c['cell_type']=='markdown' for c in nb['cells'])
(ROOT/'_data/catalog.json').write_text(json.dumps(items,ensure_ascii=False,indent=2)+'\n')

intro='''# AMD ROCm AI 开发者教程（中文翻译项目）

本项目将 AMD **AI Developer Hub** 的教程整理为中文阅读资料，内容覆盖推理、微调、预训练和 GPU 开发与优化。原文以 Jupyter Notebook 发布，译文同时提供 Markdown、中英对照和中文 Notebook。

- 原站：[AI Developer Hub](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/index.html)
- 原文仓库：[ROCm/gpuaidev](https://github.com/ROCm/gpuaidev)
- 项目维护者：[GoBang5](https://github.com/GoBang5)
- 组织方式参考：[xindoo/agentic-design-patterns](https://github.com/xindoo/agentic-design-patterns)

> **当前版本：42 篇机器初译。** 已统一核心术语并进行结构与代码一致性检查，尚待逐篇人工校对。代码、命令、提示词和运行输出保留原文；尚未在 AMD GPU 上验证。使用教程时请同时查看对应原文和版本要求。

## 🌐 在线阅读

**[进入中文阅读站](https://GoBang5.github.io/rocm-ai-developer-hub-zh/)** · [完整目录](SUMMARY.md) · [教程筛选](tutorial-selector.md) · [翻译进度](progress.md)

## 📚 教程目录

'''
readme='![中文教程封面](images/cover.svg)\n\n'+intro
summary='# 目录\n\n- [项目简介](README.md)\n- [教程筛选](tutorial-selector.md)\n- [术语表](glossary.md)\n\n'
progress='''# 翻译进度

同步日期：2026-10-12。上游文档版本：16.0。固定提交：`0071bfd4bd3bd9525405c09a6ae69c983b09312a`。

| 分类 | 教程数 | 机器初译 | 全文人工校对 | AMD GPU 运行验证 |
| --- | ---: | ---: | ---: | ---: |
| 推理 | 20 | 20 | 0 | 0 |
| 微调 | 6 | 6 | 0 | 0 |
| 预训练 | 9 | 9 | 0 | 0 |
| GPU 开发与优化 | 7 | 7 | 0 | 0 |
| 合计 | 42 | 42 | 0 | 0 |

“机器初译”表示全部 Markdown 说明单元已翻译，代码保持原样；不代表逐篇人工审核完成。“人工校对”需要核对全文技术含义、术语、链接和格式。“运行验证”需要记录实际硬件、软件版本与运行结果。

| 教程 | 译文 | 状态 |
| --- | --- | --- |
'''
levels={'beginner':'入门','intermediate':'进阶','advanced':'高级','unknown':'未标注'}
select='# 教程筛选\n\n选择知识水平和教程类型，找到适合自己的教程。在线阅读站提供交互筛选；下面的知识水平来自原文或官方选择器。\n\n| 教程 | 分类 | 知识水平 |\n| --- | --- | --- |\n'
for cat in ['inference','fine_tune','pretrain','gpu_dev_optimize']:
 rows=[it for it in items if it['category']==cat];label=rows[0]['category_zh'];readme+=f'### {label}（{len(rows)} 篇）\n\n';summary+=f'## {label}\n\n';idx=f'# {label}教程\n\n'
 for it in rows:
  path=f'chapters/{cat}/{it["slug"]}.md';line=f'{it["number"]}. [{it["title"]}]({path})\n'
  readme+=line;summary+=line;idx+=f'- [{it["title"]}]({it["slug"]}.md)\n'
  progress+=f'| {it["title"]} | [中文]({path}) | 机器初译，待校对 |\n'
  select+=f'| [{it["title"]}]({path}) | {label} | {levels[it["level"]]} |\n'
 readme+='\n';summary+='\n'
 p=ROOT/'chapters'/cat/'README.md';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(idx)
readme+='''## 📁 项目结构

```text
rocm-ai-developer-hub-zh/
├── README.md                 # 项目说明与中文目录
├── SUMMARY.md                # 完整阅读目录
├── chapters/                 # 中文 Markdown 教程
├── original/                 # 固定版本英文 Notebook、Markdown 与图片
├── notebooks/                # 中文 Notebook（代码保持原样）
├── bilingual/                # 按单元对齐的中英对照
├── images/                   # 本项目封面
├── glossary.md               # 中英术语表
├── translation-guide.md      # 翻译与校对规范
├── progress.md               # 初译、校对与验证进度
├── _data/catalog.json        # 来源、版本、作者、难度与状态
└── scripts/                  # 翻译、导出、验证和阅读站构建
```

## 🤝 参与校对

欢迎通过 Issue 报告错误，或提交 PR 改进译文。请先阅读 [贡献指南](CONTRIBUTING.md) 和 [翻译规范](translation-guide.md)，并使用 [术语表](glossary.md)。

## 🛠️ 本地构建

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/validate.py
python scripts/build_site.py
python -m http.server 8000 --directory _site
```

阅读站由 GitHub Actions 构建并部署到 GitHub Pages。翻译缓存保存在任务工作目录，仓库不包含密钥或用户凭据。
'''
for filename,text in [('README.md',readme),('SUMMARY.md',summary),('progress.md',progress),('tutorial-selector.md',select),('overview.md',intro)]: (ROOT/filename).write_text(text)
(ROOT/'NOTICE.md').write_text('''# 来源说明

原文项目：[AMD ROCm AI Developer Hub](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/index.html)。源文件仓库：[ROCm/gpuaidev](https://github.com/ROCm/gpuaidev)。同步日期：2026-10-12，文档版本 16.0，固定提交 `0071bfd4bd3bd9525405c09a6ae69c983b09312a`。

原文版权归 Advanced Micro Devices, Inc. 及各篇标注作者所有。AMD、ROCm、Instinct 等商标归各权利人所有。本项目不复制官方标识作为自己的标识。

项目结构参考 xindoo/agentic-design-patterns，仅借鉴目录组织方式；README、封面、阅读站样式和脚本为本项目自行编写，没有复制其网站模板或个人介绍。

初译由 Google Translate 公共接口生成，使用保护片段保留代码、URL、HTML 与公式，并统一技术术语。发布前进行了结构验证与抽样检查；全文人工校对和实际硬件验证的状态独立记录于 progress.md。

原文中涉及的第三方模型、软件、数据集和图片仍适用各自许可。运行代码中的提示词、注释和输出保留英语。已有原文中的技术问题不以静默修改代码的方式处理。
''')
(ROOT/'translation-guide.md').write_text('''# 翻译与校对规范

1. 保持原文标题层级、段落、表格、列表和步骤顺序；不将摘要当作全文译文。
2. 术语采用 glossary.md，框架名、模型名、产品名保留官方拼写。首次出现缩写时可补充中文解释。
3. 不翻译或修改代码、Shell 命令、配置键、文件名、环境变量、API 名、模型 ID、代码里的提示词与运行输出。
4. 保留作者、原文链接和固定上游提交。译者补充内容须用“译注”标识。
5. 数值、单位、GPU 型号、版本号、公式与性能结果必须逐项对照原文。
6. 将性能分析译为“性能分析”，kernel 译为“内核”，token 依语境译为“Token”；避免将 Triton、slime 等框架名当作普通词翻译。
7. 修改 Notebook 的 Markdown 单元后，同步 chapters/ 与 bilingual/，运行 python scripts/validate.py 和 python scripts/build_site.py。
8. 只有人工逐篇核对全文后才将 catalog 中的 reviewed 改为 true，并更新 progress.md；机器初译不能计入人工校对。
9. 运行验证需记录硬件、ROCm、框架版本和结果。没有运行证据时 runtime_verified 保持 false。

更新上游时先比较旧版与新版，以固定提交为单位同步；避免直接使用 latest 覆盖旧内容而失去来源记录。translate_notebooks.py 使用外部翻译服务，生成的是待校对草稿。未修改的原文 Notebook 作为验证基准，不应在本仓库中编辑。
''')
(ROOT/'CONTRIBUTING.md').write_text('''# 贡献指南

欢迎修正错译、统一术语、改善排版与补充验证记录。

1. Fork 本仓库，创建分支。
2. 在 Issue 或 PR 中说明教程名称、原文段落、问题及修正理由。
3. 修改中文 Notebook 的 Markdown 单元，并同步中文 Markdown 和中英对照；代码改动应作为独立问题处理。
4. 运行 python scripts/validate.py 与 python scripts/build_site.py。
5. 提交 PR。涉及完整校对或硬件验证时，附上可核查的记录，再更新相应状态。

请勿提交 API 密钥、访问令牌、个人数据或未核实的性能数字。
''')
(ROOT/'PROJECT_STRUCTURE.md').write_text('''# 项目结构说明

chapters/、original/、notebooks/、bilingual/ 使用相同的分类和文件名，便于一一对照。original/ 中的 .ipynb 是上游固定版本的原始文件；.md 是按原始单元导出的阅读版本。图片统一保存在 original/assets/，内嵌图片按内容哈希提取，避免失效的超长 data URI。

_data/catalog.json 保存全部 42 篇教程的标题、作者、分类、难度、来源、原文哈希和翻译状态。scripts/validate.py 检查覆盖率、原文完整性、代码和命令一致性、保护片段、图片与链接。scripts/build_site.py 生成静态阅读页面，不执行任何教程代码。

scripts/translate_notebooks.py 需要 --upstream 与 --cache 参数；上游源码和翻译缓存保存在独立任务目录，不将个人机器绝对路径写进仓库。scripts/export_docs.py 与 scripts/project_docs.py 分别导出教程与生成目录，重新运行会覆盖生成内容，修改前应保存人工校对成果。
''')
print('Generated project documents and metadata for',len(items),'tutorials')
