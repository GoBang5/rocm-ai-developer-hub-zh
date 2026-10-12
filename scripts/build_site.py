#!/usr/bin/env python3
"""Build a static reading site. Never executes notebook code."""
from pathlib import Path
import html,json,re,shutil,urllib.parse
import markdown
from pygments.formatters import HtmlFormatter
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'_site';REPO='https://github.com/GoBang5/rocm-ai-developer-hub-zh'
items=json.loads((ROOT/'_data/catalog.json').read_text())
if OUT.exists():shutil.rmtree(OUT)
OUT.mkdir();(OUT/'.nojekyll').touch()

def route(path):
 return 'index.html' if path=='README.md' else path.replace('.md','.html')

def render(text,current):
 md=markdown.Markdown(extensions=['tables','toc','md_in_html','pymdownx.superfences','pymdownx.arithmatex'],extension_configs={'pymdownx.arithmatex':{'generic':True}})
 body=md.convert(text)
 def fix(m):
  attr,url=m[1],html.unescape(m[2]);parsed=urllib.parse.urlsplit(url)
  if parsed.scheme or url.startswith(('#','//')):return m[0]
  if parsed.path.endswith('.md'):
   import posixpath
   target=posixpath.normpath(posixpath.join(str(Path(current).parent),parsed.path))
   url=url.replace('README.md','index.html') if target=='README.md' else url.replace('.md','.html')
  if parsed.path.endswith('.ipynb'):
   import posixpath
   target=posixpath.normpath(posixpath.join(str(Path(current).parent),parsed.path))
   url=REPO+'/blob/main/'+target
  return attr+'="'+html.escape(url,quote=True)+'"'
 body=re.sub(r'(href|src)="([^"]+)"',fix,body)
 return body,md.toc

def nav(prefix):
 s=f'<a class="nav-home" href="{prefix}index.html">项目首页</a><a href="{prefix}tutorial-selector.html">教程筛选</a>'
 for cat in ['inference','fine_tune','pretrain','gpu_dev_optimize']:
  rows=[x for x in items if x['category']==cat];s+=f'<details open><summary>{rows[0]["category_zh"]}<span>{len(rows)}</span></summary><div>'
  for it in rows:
   s+=f'<a data-title="{html.escape(it["title"]+" "+it["title_en"],quote=True)}" href="{prefix}chapters/{cat}/{it["slug"]}.html"><small>{it["number"]:02}</small>{html.escape(it["title"])}</a>'
  s+='</div></details>'
 s+=f'<div class="nav-tools"><a href="{prefix}glossary.html">术语表</a><a href="{prefix}progress.html">翻译进度</a><a href="{prefix}changelog.html">上游更新记录</a><a href="{prefix}translation-guide.html">翻译规范</a><a href="{prefix}NOTICE.html">来源与授权</a><a href="{prefix}licensing.html">许可与支持</a></div>'
 return s

def page(path,title,body,toc=''):
 depth=len(Path(path).parent.parts);prefix='../'*depth
 active=next((x for x in items if path==f'chapters/{x["category"]}/{x["slug"]}.md'),None)
 context=''
 if active:
  rel=active['category']+'/'+active['slug'];context=f'<div class="chapter-meta"><span>{html.escape(active["category_zh"])}</span><span>机器初译 · 待校对</span><a href="{prefix}original/{rel}.html">英文原文</a><a href="{prefix}bilingual/{rel}.html">中英对照</a><a href="{REPO}/blob/main/notebooks/{rel}.ipynb">中文 Notebook ↗</a></div>'
  i=active['number']-1;links=[]
  for j,label in [(i-1,'← 上一篇'),(i+1,'下一篇 →')]:
   if 0<=j<len(items):it=items[j];links.append(f'<a href="{prefix}chapters/{it["category"]}/{it["slug"]}.html">{label}<strong>{html.escape(it["title"])}</strong></a>')
  body+='<nav class="page-turn">'+''.join(links)+'</nav>'
 payload='''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>'''+html.escape(title)+''' · ROCm AI 中文教程</title>
<meta name="description" content="AMD ROCm AI Developer Hub 中文教程：推理、微调、预训练与 GPU 优化">
<link rel="stylesheet" href="'''+prefix+'''site.css">
<script>try{document.documentElement.dataset.theme=localStorage.getItem('rocm-theme')||'light'}catch(e){}</script>
<script>window.MathJax={tex:{inlineMath:[['$','$'],['\\\\(','\\\\)']]},options:{skipHtmlTags:['script','noscript','style','textarea','pre','code']}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js">
</script>
</head>
<body>
<a class="skip" href="#content">跳到正文</a>
<header>
<button id="menu" aria-label="展开导航" aria-expanded="false">☰</button>
<a class="brand" href="'''+prefix+'''index.html">
<span class="brand-symbol">R</span>
<span>ROCm <b>AI 开发者教程</b>
<small>中文翻译项目</small>
</span>
</a>
<div class="header-right">
<a href="'''+REPO+'''">GitHub ↗</a>
<button id="theme" aria-label="切换明暗主题">◐</button>
</div>
</header>
<aside class="sidebar">
<label class="searchbox">
<span>⌕</span>
<input id="search" type="search" placeholder="搜索教程 /" aria-label="搜索教程">
</label>
<div id="search-empty" hidden>没有匹配的教程</div>
<nav>'''+nav(prefix)+'''</nav>
<footer>上游 v16.0 · 2026-10-12<br>42 篇初译 · 逐篇校对中</footer>
</aside>
<div id="shade">
</div>
<div class="reading-grid">
<main id="content">'''+context+body+'''<footer class="article-footer">原文版权归 AMD 及各作者所有 · 独立中文翻译项目<br>
<a href="'''+REPO+'/edit/main/'+path+'''">改进本页 ↗</a> · <a href="'''+prefix+'''NOTICE.html">来源与授权</a>
</footer>
</main>
<aside class="page-toc">
<span>本页目录</span>'''+toc+'''</aside>
</div>
<script src="'''+prefix+'''site.js">
</script>
</body>
</html>'''
 dest=OUT/route(path);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(payload)

for folder in ['chapters','original','bilingual']:
 for p in (ROOT/folder).rglob('*.md'):
  text=p.read_text();title=next(iter(re.findall(r'(?m)^#\s+(.+)',text)),p.stem);body,toc=render(text,p.relative_to(ROOT).as_posix());page(p.relative_to(ROOT).as_posix(),title,body,toc)
for p in ROOT.glob('*.md'):
 text=p.read_text();title=next(iter(re.findall(r'(?m)^#\s+(.+)',text)),p.stem);body,toc=render(text,p.name)
 if p.name=='tutorial-selector.md':
  body='''<p class="eyebrow">FIND YOUR NEXT TUTORIAL</p><h1>找到你的下一篇教程</h1><p>按知识水平和技术方向筛选，从适合自己的教程开始。</p><div class="filters"><label>知识水平<select id="level"><option value="all">全部水平</option><option value="beginner">入门</option><option value="intermediate">进阶</option><option value="advanced">高级</option></select></label><label>教程类型<select id="category"><option value="all">全部类型</option><option value="inference">推理</option><option value="fine_tune">微调</option><option value="pretrain">预训练</option><option value="gpu_dev_optimize">GPU 开发与优化</option></select></label></div><p id="filter-count"></p><div class="tutorial-grid">'''
  for it in items:
   lv={'beginner':'入门','intermediate':'进阶','advanced':'高级','unknown':'未标注'}[it['level']]
   body+=f'<a class="tutorial-card" data-category="{it["category"]}" data-level="{it["level"]}" href="chapters/{it["category"]}/{it["slug"]}.html"><span>{it["category_zh"]} · {lv}</span><h2>{it["title"]}</h2><p>{it["title_en"]}</p><b>阅读教程 →</b></a>'
  body+='</div>';toc=''
 page(p.name,title,body,toc)
shutil.copytree(ROOT/'original/assets',OUT/'original/assets',dirs_exist_ok=True);shutil.copytree(ROOT/'images',OUT/'images',dirs_exist_ok=True)
for asset in ['site.css','site.js']:shutil.copy2(ROOT/asset,OUT/asset)
(OUT/'syntax.css').write_text(HtmlFormatter(style='friendly').get_style_defs('.codehilite'))
print('Built',len(list(OUT.rglob('*.html'))),'HTML pages')
