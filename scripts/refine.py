"""Apply documented terminology and attribution fixes to Markdown cells only."""
from pathlib import Path
import json,hashlib,re
ROOT=Path(__file__).resolve().parents[1];items=json.loads((ROOT/'_data/catalog.json').read_text());revisions=json.loads((ROOT/'_data/revisions.json').read_text())
FENCE=re.compile(r'(?ms)^\s*(`{3,}|~{3,})[^\n]*\n.*?^\s*\1[^\n]*(?:\n|$)')
fixes={'火炬泰坦':'TorchTitan','夸克':'Quark','火炬 FSDP':'torch FSDP','推测的 草稿模型':'投机解码草稿模型','使用_torch_fsdp2':'use_torch_fsdp2','性能剖析':'性能分析','性能剖面':'性能分析','谷歌 ADK':'Google ADK','拥抱脸部':'Hugging Face','拥抱脸':'Hugging Face','笔记本电脑':'Notebook','笔记本':'Notebook','8 波乒乓':'8 个 wavefront 乒乓','挑战扩大智能体':'挑战：扩展智能体能力','Airbnb 发现者':'Airbnb 房源查找助手'}
for it in items:
 rel=Path(it['category'])/(it['slug']+'.ipynb');en=json.loads((ROOT/'original'/rel).read_text());zp=ROOT/'notebooks'/rel;zh=json.loads(zp.read_text())
 for ec,zc in zip(en['cells'],zh['cells']):
  if ec['cell_type']!='markdown':continue
  es=''.join(ec['source']);h=hashlib.sha256(es.encode()).hexdigest();zs=revisions.get(h,''.join(zc['source']))
  def refine(s):
   spans=[]
   def save(m):spans.append(m[0]);return f'RFX{len(spans)-1:05}XFR'
   s=FENCE.sub(save,s);s=re.sub(r'`[^`\n]*`',save,s)
   for wrong,right in fixes.items():s=s.replace(wrong,right)
   if re.search(r'\bagents?\b',es,re.I) and not re.search(r'\bproxy\b',es,re.I):
    s=s.replace('代理','智能体')
   if it['category']=='gpu_dev_optimize' and re.search(r'\btiles?\b',es,re.I):
    s=s.replace('瓦片','分块');s=re.sub(r'(\d+\s*[×x]\s*\d+)瓦',r'\1分块',s)
   if it['category']=='gpu_dev_optimize' and re.search(r'\bthreads?\b',es,re.I):
    s=re.sub(r'(\d+)线(?!程)',r'\1线程',s)
   for i,span in enumerate(spans):s=s.replace(f'RFX{i:05}XFR',span)
   return s
  zs=refine(zs)
  author=re.search(r'(?mi)^\s*\*\*Authors?:?\*\*:?\s*([^\n]+)',es)
  if author:
   name=author[1].rstrip();it['author']=name.strip()
   zs=re.sub(r'(?mi)^\s*\*\*作者[:：]?\*\*[:：]?\s*[^\n]+','**作者**：'+name,zs)
  authored=re.search(r'(?mi)^\s*\*\*Authored by\*\*:?\s*([^\n]+)',es)
  if authored:
   credit=authored[1].strip()
   credit=credit.replace(' and modified by ', '；由 ').replace(' to run on AMD GPUs.', ' 修改以支持 AMD GPU。')
   it['author']=re.sub(r'\[([^\]]+)\]\([^)]+\)',r'\1',credit)
   zs=re.sub(r'(?mi)^\s*\*\*作者[:：]?\*\*[:：]?\s*[^\n]+','**作者**：'+credit,zs)
  coauthor=re.search(r'(?mi)^\s*\*\*Co-authors?:?\*\*:?\s*([^\n]+)',es)
  if coauthor:
   zs=re.sub(r'(?mi)^\s*\*\*合著者[:：]?\*\*[:：]?\s*[^\n]+','**合著者**：'+coauthor[1].rstrip(),zs)
  level=re.search(r'(?mi)^\s*\*\*Knowledge level:?\*\*:?\s*(\w+)',es)
  if level:it['level']=level[1].lower()
  # Keep links inside Markdown using ASCII punctuation and preserve original button labels where needed.
  zs=re.sub(r'\]（([^\n）]+)）',r'](\1)',zs)
  zs=re.sub(r'【([^\n】]+)】[（(]([^\n)）]+)[）)]',r'[\1](\2)',zs)
  zs=re.sub(r'(?m)^(\s*\d+[.)])\s*(?=\[)',r'\1 ',zs)
  zc['source']=zs.splitlines(keepends=True)
 if zh['cells'] and zh['cells'][0]['cell_type']=='markdown':
  first=''.join(zh['cells'][0]['source']);first=re.sub(r'(?m)^#\s+[^\n]+','# '+it['title'],first,count=1);zh['cells'][0]['source']=first.splitlines(keepends=True)
 zp.write_text(json.dumps(zh,ensure_ascii=False,indent=1)+'\n')
(ROOT/'_data/catalog.json').write_text(json.dumps(items,ensure_ascii=False,indent=2)+'\n')
print('Applied terminology, title, attribution fixes and one manual passage translation to 42 notebooks')
