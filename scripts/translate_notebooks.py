#!/usr/bin/env python3
"""Translate Markdown cells with code, URLs, math and HTML protected.
Uses Google Translate's public gtx endpoint; produces machine drafts, not reviewed text.
No credentials required. Cache lives outside the repository (pass --cache).
"""
import argparse, concurrent.futures, hashlib, json, re, shutil, threading, time
import urllib.parse, urllib.request
from pathlib import Path

GLOSSARY = {
 'speculative decoding': '投机解码', 'prefill': '预填充',
 'PD disaggregation': '预填充与解码分离', 'multi-token prediction': '多 Token 预测',
 'fine-tuning': '微调', 'pretraining': '预训练', 'inference': '推理',
 'large language model': '大语言模型', 'vision-language model': '视觉语言模型',
 'retrieval-augmented generation': '检索增强生成',
 'model context protocol': '模型上下文协议',
 'knowledge level': '知识水平', 'prerequisites': '前置条件',
 'throughput': '吞吐量', 'latency': '延迟', 'batch size': '批大小',
 'tensor parallelism': '张量并行', 'pipeline parallelism': '流水线并行',
 'data parallelism': '数据并行', 'mixed precision': '混合精度',
 'gradient accumulation': '梯度累积', 'learning rate': '学习率',
 'checkpoint': '检查点', 'quantization': '量化', 'GPU kernel': 'GPU 内核',
 'draft model': '草稿模型', 'target model': '目标模型',
 'reinforcement learning': '强化学习', 'reward model': '奖励模型',
 'time to first token': '首 Token 延迟', 'inter-token latency': 'Token 间延迟',
 'key-value cache': '键值缓存', 'KV cache': 'KV 缓存',
 'mixture of experts': '混合专家', 'parameter-efficient fine-tuning': '参数高效微调',
 'group relative policy optimization': '组相对策略优化',
}
# Code blocks, inline code, LaTeX, HTML tags and all URL targets are restored byte for byte.
PROTECT = re.compile(
 r'(?ms)^\s*(`{3,}|~{3,})[^\n]*\n.*?^\s*\1[^\n]*(?:\n|$)'
 r'|`+[^`\n]*`+'
 r'|\$\$.*?\$\$|(?<!\\)\$[^$\n]+?(?<!\\)\$'
 r'|\\\[.*?\\\]|\\\(.*?\\\)'
 r'|<!--[\s\S]*?-->|</?[A-Za-z][^>]*>'
 r'|(?<=\]\()[^\s)\n]+'
 r'|(?:https?://|data:image/|attachment:|\.\./)[^\s<>"\]]+'
 r'|(?<=\]\()#[^\s)]+'
 r'|^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$'
)
LOCK=threading.Lock()

def mask(text):
 values=[]
 def save(m):
  s=m.group(0)
  # A link URL regex may include the closing Markdown parenthesis.
  trail=''
  if s.startswith(('http','../','data:image/','attachment:')):
   while s.endswith((')', '.', ',')):
    if s.endswith(')') and s.count('(')>=s.count(')'): break
    trail=s[-1]+trail; s=s[:-1]
  key=f'ZXQ{len(values):05d}QXZ'; values.append(s); return key+trail
 text=PROTECT.sub(save,text)
 # Author names are attribution and must not be translated.
 text=re.sub(r'(?mi)^(\s*\*?\*?Authors?\*?\*?\s*:\s*)([^\n]+)',lambda m:m[1]+save(type('M',(),{'group':lambda self,n:m[2]})()),text)
 for en,zh in sorted(GLOSSARY.items(),key=lambda p:-len(p[0])):
  def term(m):
   key=f'ZXQ{len(values):05d}QXZ'; values.append(zh); return key
  text=re.sub(r'(?i)\b'+re.escape(en)+r'\b',term,text)
 return text,values

def unmask(text,values):
 for i,v in enumerate(values):
  key=f'ZXQ{i:05d}QXZ'
  text=re.sub(r'ZXQ\s*'+f'{i:05d}'+r'\s*QXZ',key,text,flags=re.I)
  if text.count(key)!=1: raise ValueError(f'Protected span {i} occurred {text.count(key)} times')
  text=text.replace(key,v)
 if re.search(r'ZXQ\s*\d+\s*QXZ',text,re.I):raise ValueError('Unrestored marker')
 return text

def request(text):
 params=urllib.parse.urlencode({'client':'gtx','sl':'en','tl':'zh-CN','dt':'t','q':text})
 req=urllib.request.Request('https://translate.googleapis.com/translate_a/single',data=params.encode(),headers={'User-Agent':'Mozilla/5.0','Content-Type':'application/x-www-form-urlencoded'})
 # GET is supported by this endpoint; POST sometimes rejects larger payloads.
 req=urllib.request.Request('https://translate.googleapis.com/translate_a/single?'+params,headers={'User-Agent':'Mozilla/5.0'})
 with urllib.request.urlopen(req,timeout=45) as r:d=json.loads(r.read())
 return ''.join(row[0] for row in d[0] if row[0])

def chunks(text,limit=3200):
 parts=re.split(r'(\n\s*\n)',text); out=[]; current=''
 for part in parts:
  if len(current)+len(part)>limit and current:out.append(current);current=''
  if len(part)>limit:
   for line in part.splitlines(keepends=True):
    if len(current)+len(line)>limit and current:out.append(current);current=''
    if len(line)>limit:
     # Do not split placeholder tokens: long prose uses punctuation/whitespace boundaries.
     words=re.split(r'(\s+)',line)
     for w in words:
      if len(current)+len(w)>limit and current:out.append(current);current=''
      current+=w
    else:current+=line
  else:current+=part
 if current:out.append(current)
 return out

def translate(src,cache):
 digest=hashlib.sha256(src.encode()).hexdigest(); path=cache/(digest+'.json')
 if path.exists():return json.loads(path.read_text())['translation']
 protected,values=mask(src)
 if not re.search(r'[A-Za-z]{2,}',re.sub(r'ZXQ\d+QXZ','',protected)):
  result=unmask(protected,values)
 else:
  failure=None
  for attempt in range(5):
   try:
    translated=''.join(request(c).rstrip('\n')+('\n' if c.endswith('\n') else '') for c in chunks(protected,limit=3200 if attempt==0 else 1000))
    result=unmask(translated,values);break
   except Exception as e:
    failure=e;time.sleep(min(2**attempt,12))
  else:
   # Preserve spans directly when a translator keeps dropping or altering markers.
   translated=[]
   for segment in re.split(r'(ZXQ\d{5}QXZ)',protected):
    if re.fullmatch(r'ZXQ\d{5}QXZ',segment) or not re.search(r'[A-Za-z]{2,}',segment):
     translated.append(segment)
    else:
     translated.extend(request(c) for c in chunks(segment,limit=1000))
   result=unmask(''.join(translated),values)
 with LOCK:
  tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps({'source':src,'translation':result},ensure_ascii=False));tmp.replace(path)
 return result

def main():
 a=argparse.ArgumentParser();a.add_argument('--upstream',required=True);a.add_argument('--cache',required=True);a.add_argument('--workers',type=int,default=5);a.add_argument('--category');args=a.parse_args()
 upstream=Path(args.upstream);root=Path(__file__).resolve().parents[1];cache=Path(args.cache);cache.mkdir(parents=True,exist_ok=True)
 sources=sorted((upstream/'docs/notebooks').rglob('*.ipynb'))
 if args.category:sources=[p for p in sources if p.parent.name==args.category]
 tasks={}; notebooks={}
 for p in sources:
  relative=p.relative_to(upstream/'docs/notebooks'); original=root/'original'/relative;original.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,original)
  nb=json.loads(p.read_text());notebooks[str(relative)]=nb
  for c in nb['cells']:
   if c['cell_type']=='markdown':
    src=''.join(c['source']);tasks.setdefault(src,None)
 print(f'Translating {len(sources)} notebooks / {len(tasks)} distinct Markdown cells',flush=True)
 failures=[];count=0
 with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
  futuremap={pool.submit(translate,src,cache):src for src in tasks}
  for future in concurrent.futures.as_completed(futuremap):
   src=futuremap[future];count+=1
   try:tasks[src]=future.result()
   except Exception as e:failures.append(str(e));print(f'FAIL {hashlib.sha256(src.encode()).hexdigest()[:10]} {e}',flush=True)
   if count%20==0 or count==len(tasks):print(f'Cells {count}/{len(tasks)}; failed {len(failures)}',flush=True)
 if failures:raise SystemExit(f'{len(failures)} cells failed; cache retained for resume.')
 for rel,nb in notebooks.items():
  for c in nb['cells']:
   if c['cell_type']=='markdown':c['source']=tasks[''.join(c['source'])].splitlines(keepends=True)
  target=root/'notebooks'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
 print(f'Wrote {len(notebooks)} translated notebooks',flush=True)
if __name__=='__main__':main()
