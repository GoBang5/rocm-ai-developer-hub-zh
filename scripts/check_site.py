#!/usr/bin/env python3
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'_site'
class Page(HTMLParser):
 def __init__(self):super().__init__();self.refs=[];self.ids=set()
 def handle_starttag(self,tag,attrs):
  d=dict(attrs)
  for k in ['id','name']:
   if d.get(k):self.ids.add(d[k])
  for k in ['href','src']:
   if d.get(k):self.refs.append((tag,k,d[k]))
pages={}
for p in OUT.rglob('*.html'):
 parser=Page();parser.feed(p.read_text());pages[p.resolve()]=parser
errors=[];anchors=[];refs=0
for p,page in pages.items():
 for tag,key,url in page.refs:
  u=urlsplit(url)
  if u.scheme or u.netloc:continue
  refs+=1;dest=(p.parent/unquote(u.path)).resolve() if u.path else p
  if dest.is_dir():dest=dest/'index.html'
  if not dest.exists():errors.append(f'{p.relative_to(OUT)} -> {url}')
  elif u.fragment and dest in pages and unquote(u.fragment) not in pages[dest].ids:anchors.append(f'{p.relative_to(OUT)} -> {url}')
report={'pages':len(pages),'local_references':refs,'missing_files':sorted(set(errors)),'missing_anchors':sorted(set(anchors))}
(ROOT/'_data/site-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v if not isinstance(v,list) else len(v) for k,v in report.items()}))
for s in sorted(set(errors))[:30]:print('FILE',s)
for s in sorted(set(anchors))[:30]:print('ANCHOR',s)
if errors or anchors:raise SystemExit(1)
