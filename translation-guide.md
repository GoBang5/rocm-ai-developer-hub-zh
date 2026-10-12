# 翻译与校对规范

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
