#!/usr/bin/env python3
"""Verify coverage, immutable originals, code preservation and protected spans."""
from pathlib import Path
import collections,hashlib,json,re,sys
ROOT=Path(__file__).resolve().parents[1]
items=json.loads((ROOT/'_data/catalog.json').read_text());errors=[];warnings=[];totals=collections.Counter()
FENCE=re.compile(r'(?ms)^\s*(`{3,}|~{3,})([^\n]*)\n(.*?)^\s*\1[^\n]*(?:\n|$)')

def fences(s):return [(m[2].strip(),m[3]) for m in FENCE.finditer(s)]

for it in items:
 rel=Path(it['category'])/(it['slug']+'.ipynb');ep=ROOT/'original'/rel;zp=ROOT/'notebooks'/rel
 paths=[ep,zp]+[ROOT/f/rel.with_suffix('.md') for f in ['chapters','original','bilingual']]
 for p in paths:
  if not p.exists():errors.append(f'Missing: {p.relative_to(ROOT)}')
 if not ep.exists() or not zp.exists():continue
 if hashlib.sha256(ep.read_bytes()).hexdigest()!=it['original_sha256']:errors.append(f'Original differs: {rel}')
 en=json.loads(ep.read_text());zh=json.loads(zp.read_text())
 if len(en['cells'])!=len(zh['cells']) or len(en['cells'])!=it['cell_count']:errors.append(f'Cell count: {rel}');continue
 for i,(e,z) in enumerate(zip(en['cells'],zh['cells'])):
  if e['cell_type']!=z['cell_type']:errors.append(f'Cell type: {rel}:{i}')
  es=''.join(e['source']);zs=''.join(z['source'])
  if e['cell_type']=='code':
   totals['code_cells']+=1
   if e['source']!=z['source']:errors.append(f'Code changed: {rel}:{i}')
  if e.get('outputs')!=z.get('outputs'):errors.append(f'Outputs changed: {rel}:{i}')
  if e['cell_type']=='markdown':
   totals['markdown_cells']+=1
   if es.strip() and not zs.strip():errors.append(f'Empty translation: {rel}:{i}')
   if re.search(r'ZXQ\s*\d+\s*QXZ',zs,re.I):errors.append(f'Marker: {rel}:{i}')
   if fences(es)!=fences(zs):errors.append(f'Markdown code fences differ: {rel}:{i}')
   # All inline code and external URL literals must remain available.
   for token in re.findall(r'`[^`\n]+`',es):
    if token not in zs:errors.append(f'Inline code lost: {rel}:{i}: {token[:60]}')
   for url in re.findall(r'https?://[^\s<>"\])`]+',es):
    if url not in zs:errors.append(f'URL lost: {rel}:{i}: {url[:80]}')
   prose=FENCE.sub('',zs);prose=re.sub(r'`[^`]+`|<[^>]+>|https?://\S+','',prose)
   enprose=FENCE.sub('',es);enprose=re.sub(r'`[^`]+`|<[^>]+>|https?://\S+|data:image/\S+','',enprose)
   if len(re.findall(r'[A-Za-z]+',enprose))>20 and not re.search(r'[\u4e00-\u9fff]',prose):warnings.append(f'No Chinese prose: {rel}:{i}')
 totals['tutorials']+=1
# Markdown assets and cross-document file links. Data URIs are extracted during export.
for folder in ['chapters','bilingual','original']:
 for p in (ROOT/folder).rglob('*.md'):
  s=p.read_text()
  if 'data:image/' in s:errors.append(f'Unextracted image: {p.relative_to(ROOT)}')
  s=FENCE.sub('',s);s=re.sub(r'`[^`\n]+`','',s)
  for url in re.findall(r'\]\(([^\n)]+)\)',s):
   url=url.strip().split(' "')[0]
   if url.startswith(('http:','https:','#','mailto:','data:')):continue
   dest=(p.parent/url.split('#')[0]).resolve()
   if not dest.exists():warnings.append(f'Upstream/local link missing: {p.relative_to(ROOT)} -> {url[:120]}')
counts=collections.Counter(it['category'] for it in items)
if dict(counts)!={'inference':20,'fine_tune':6,'pretrain':9,'gpu_dev_optimize':7}:errors.append(f'Coverage count: {dict(counts)}')
report={'errors':errors,'warnings':sorted(set(warnings)),'counts':dict(totals),'source_commit':items[0]['upstream_sha'],'machine_draft_tutorials':sum(i['status']=='machine_draft' for i in items),'reviewed_tutorials':sum(i['reviewed'] for i in items),'runtime_verified_tutorials':sum(i['runtime_verified'] for i in items)}
(ROOT/'_data/validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'counts':dict(totals),'errors':len(errors),'warnings':len(set(warnings))},ensure_ascii=False))
for e in errors[:40]:print('ERROR',e)
for w in sorted(set(warnings))[:30]:print('WARN',w)
if errors:sys.exit(1)
