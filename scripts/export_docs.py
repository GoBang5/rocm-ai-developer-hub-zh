#!/usr/bin/env python3
"""Export originals, Chinese notebooks, Markdown chapters and cell-aligned bilingual files."""
import base64, hashlib, html, json, re, shutil, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def extract_images(text,asset_prefix):
 def save(m):
  data=base64.b64decode(m[2]);suffix={'jpeg':'jpg','png':'png','gif':'gif','webp':'webp','svg+xml':'svg'}.get(m[1],'bin')
  name='embedded-'+hashlib.sha256(data).hexdigest()[:20]+'.'+suffix
  (ROOT/'original/assets'/name).write_bytes(data)
  return asset_prefix+name
 return re.sub(r'data:image/([^;]+);base64,([A-Za-z0-9+/=]+)',save,text)


def relocate(text,prefix):
 text=extract_images(text,prefix)
 return text.replace('../assets/',prefix)


def slug(text):
 # English heading anchors survive heading translation and match GitHub's common slug rules.
 text=re.sub(r'<[^>]*>|`|\*','',text).lower()
 text=re.sub(r'[^\w\s-]','',text).strip().replace(' ','-')
 return text


def preserve_anchors(en,zh):
 enheads=re.findall(r'(?m)^#{1,6}\s+(.+?)\s*#*$',en)
 count=0;seen={}
 def anchor(m):
  nonlocal count
  if count>=len(enheads):return m[0]
  key=slug(enheads[count]);count+=1
  n=seen.get(key,0);seen[key]=n+1
  if n:key+=f'-{n}'
  return '<a id="'+html.escape(key,quote=True)+'"></a>\n\n'+m[0]
 return re.sub(r'(?m)^#{1,6}\s+.+?\s*#*$',anchor,zh)


def markdown_cell(c,prefix):
 text=''.join(c['source'])
 if c['cell_type']=='markdown':return relocate(text,prefix).strip()
 code=text.rstrip('\n');longest=max([len(x) for x in re.findall(r'`+',code)] or [2]);fence='`'*max(3,longest+1)
 return fence+'python\n'+code+'\n'+fence


def outputs(c,prefix):
 blocks=[]
 for out in c.get('outputs',[]):
  text=out.get('text') or out.get('data',{}).get('text/plain')
  if text:
   t=''.join(text) if isinstance(text,list) else text
   t=re.sub(r'\x1b\[[0-9;]*[A-Za-z]','',t)
   blocks.append('```text\n'+t.rstrip('\n')+'\n```')
  for mime,suffix in [('image/png','png'),('image/jpeg','jpg')]:
   data=out.get('data',{}).get(mime)
   if data:
    data=''.join(data) if isinstance(data,list) else data
    image=base64.b64decode(data);name='output-'+hashlib.sha256(image).hexdigest()[:20]+'.'+suffix
    (ROOT/'original/assets'/name).write_bytes(image)
    blocks.append(f'![原文运行输出]({prefix}{name})')
 return '\n\n'.join(blocks)


def main():
 upstream=Path(sys.argv[1]);catalog=json.loads((ROOT/'_data/catalog.json').read_text());assets=ROOT/'original/assets'
 shutil.copytree(upstream/'docs/notebooks/assets',assets,dirs_exist_ok=True)
 for item in catalog:
  rel=Path(item['category'])/(item['slug']+'.ipynb');ep=ROOT/'original'/rel;zp=ROOT/'notebooks'/rel
  en=json.loads(ep.read_text());zh=json.loads(zp.read_text());assert len(en['cells'])==len(zh['cells'])
  # Keep translated notebooks portable. English backups remain byte-for-byte upstream originals.
  for c in zh['cells']:
   if c['cell_type']=='markdown':c['source']=relocate(''.join(c['source']),'../../original/assets/').splitlines(keepends=True)
  zp.write_text(json.dumps(zh,ensure_ascii=False,indent=1)+'\n')
  enblocks=[];zhblocks=[];pairs=[]
  for n,(ec,zc) in enumerate(zip(en['cells'],zh['cells']),1):
   e=markdown_cell(ec,'../assets/');z=markdown_cell(zc,'../../original/assets/')
   if ec['cell_type']=='markdown':
    # Curated tutorial title and stable source heading IDs.
    if not zhblocks:z=re.sub(r'(?m)^#\s+[^\n]+','# '+item['title'],z,count=1)
    z=preserve_anchors(''.join(ec['source']),z)
    pairs.extend([f'<!-- cell:{n} -->',z,'<details markdown="1">\n<summary>查看英文原文 · 单元 '+str(n)+'</summary>\n\n'+markdown_cell(ec,'../../original/assets/')+'\n\n</details>'])
   else:
    assert ec['source']==zc['source'], f'Code changed in {rel} cell {n}'
    pairs.append(z)
   out=outputs(ec,'../../original/assets/')
   if out:z+='\n\n'+out;e+='\n\n'+out.replace('../../original/assets/','../assets/');pairs.append(out)
   enblocks.append(e);zhblocks.append(z)
  info=f'> 原文作者：{item["author"]}  \n> [AMD 官方教程]({item["official_url"]}) · [固定版本源码]({item["source_url"]})  \n> 上游版本：`{item["upstream_sha"]}` · 同步日期：2026-10-12  \n> 翻译状态：机器初译，待逐篇人工校对；代码与命令保持原文，未在 AMD GPU 上运行验证。\n\n'
  links=f'[中文 Notebook](../../notebooks/{rel.as_posix()}) · [英文原文](../../original/{rel.with_suffix(".md").as_posix()}) · [中英对照](../../bilingual/{rel.with_suffix(".md").as_posix()})\n\n---\n\n'
  for folder,blocks,header in [('chapters',zhblocks,info+links),('original',enblocks,f'> [Upstream source]({item["source_url"]}) · Snapshot: `{item["upstream_sha"]}`\n\n'),('bilingual',pairs,info)]:
   path=ROOT/folder/rel.with_suffix('.md');path.parent.mkdir(parents=True,exist_ok=True)
   path.write_text(header+'\n\n'.join(blocks)+'\n')
 # General site pages use separately translated/curated Markdown below.
 print(f'Exported {len(catalog)} chapters, {len(catalog)} English originals, {len(catalog)} bilingual files, {len(catalog)} Chinese notebooks.')
if __name__=='__main__':main()
