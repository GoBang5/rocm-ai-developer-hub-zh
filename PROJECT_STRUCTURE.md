# 项目结构说明

chapters/、original/、notebooks/、bilingual/ 使用相同的分类和文件名，便于一一对照。original/ 中的 .ipynb 是上游固定版本的原始文件；.md 是按原始单元导出的阅读版本。图片统一保存在 original/assets/，内嵌图片按内容哈希提取，避免失效的超长 data URI。

_data/catalog.json 保存全部 42 篇教程的标题、作者、分类、难度、来源、原文哈希和翻译状态。scripts/validate.py 检查覆盖率、原文完整性、代码和命令一致性、保护片段、图片与链接。scripts/build_site.py 生成静态阅读页面，不执行任何教程代码。

scripts/translate_notebooks.py 需要 --upstream 与 --cache 参数；上游源码和翻译缓存保存在独立任务目录，不将个人机器绝对路径写进仓库。scripts/export_docs.py 与 scripts/project_docs.py 分别导出教程与生成目录，重新运行会覆盖生成内容，修改前应保存人工校对成果。
