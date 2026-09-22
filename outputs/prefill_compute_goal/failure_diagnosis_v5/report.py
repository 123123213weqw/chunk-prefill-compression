"""Re-audit all expected raw records; no GPU required."""
import argparse,hashlib,json,math
from pathlib import Path
from scoring import strict_score
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def report(out):
 manifest=json.loads((out/'manifest.json').read_text());prep=json.loads((ROOT/'prepared.json').read_text());selection=json.loads((ROOT/'selection.json').read_text());queue=prep['queue']
 assert json.loads((out/'queue.json').read_text())==queue
 for name,expected in manifest['source_sha256'].items():assert sha(ROOT/name)==expected,name
 assert prep['selection_sha256']==sha(ROOT/'selection.json')
 files=list((out/'units').glob('*.json'));assert len(files)==len(queue)==manifest['expected_units']
 units={p.stem:json.loads(p.read_text()) for p in files};items={x['row']['sample_id']:x for x in selection['samples']};groups={}
 for job in queue:
  u=units[job['unit_key']];assert u['status']=='COMPLETED' and u['job']==job and u['shape']['status']=='PASS'
  row=items[u['sample_id']]['row']
  for name,p in u['probes'].items():
   gold=row['gold'] if name=='task' else {'label':'ready-blue'};assert p['gold']==gold
   s=strict_score(p['answer'],gold)
   for k,v in s.items():assert p[k]==v
  assert u['probes']['control']['correct'];assert math.isfinite(u['likelihood']['log_likelihood_margin'])
  if job['method'] in ('full','compressed'):
   assert u['historical_reproduction']=='PASS'
   ref=items[u['sample_id']]['historical'][('full' if job['method']=='full' else 'd30_mean16')+'_b1024']
   for name in ('task','control'):assert u['probes'][name]['ids']==ref['probes'][name]['ids']
   if job['method']=='compressed':assert u['legacy_engine_equivalence']=='PASS'
   else:
    for g in u['gates']:assert g['ids_equal'] and g['kv_rms']<=.005 and g['logit_rms']<=.005 and g['kl']<=.001
  if job['method']!='full':
   keeps=[job['chunk_keeps'].get(str(i),16) for i in range(row['target_tokens']//64)];assert [r['keep'] for r in u['routes']]==keeps
   assert u['shape']['saved_T_layer_tokens']==sum(6*(64-k) for k in keeps)
   if job['method']=='intervention':
    key=(u['sample_id'],job['group'],next(iter(job['chunk_keeps'].values())));groups.setdefault(key,{})[job['kind']]=u
 for _,pair in groups.items():assert pair['evidence']['shape']['saved_T_layer_tokens']==pair['placebo']['shape']['saved_T_layer_tokens']
 cases=[];lines=['# v5 局部恢复诊断结果','',f'审计 PASS；完成 {len(units)}/{len(queue)} 单元；3个失败案例 + 3个正确对照。','', '这是答案可见、事后选样的机制诊断，不是新验证集准确率。所有位置为 T 内从0开始的 chunk 编号。','']
 for item in selection['samples']:
  sid=item['row']['sample_id'];loc=next(x for x in prep['locations'] if x['sample_id']==sid);base=units[sid+'__compressed'];full=units[sid+'__full']
  lines += [f"## {sid}（{item['role']}）",'',f"Gold: {json.dumps(item['row']['gold'],ensure_ascii=False)}",f"Full: {full['probes']['task']['answer'].strip().replace(chr(10),' ')}",f"压缩: {base['probes']['task']['answer'].strip().replace(chr(10),' ')}",f"仓库证据 chunks {loc['evidence']['warehouse']['sentence']['chunks']}；箱数证据 chunks {loc['evidence']['boxes']['sentence']['chunks']}。",'', '| 范围 | keep | 证据chunk | 正确 | placebo chunk | 正确 | 证据/对照正确-vs-foil logP差 |','|---|---:|---|---|---|---|---|']
  rows=[]
  for (s,g,k),pair in groups.items():
   if s!=sid:continue
   e,p=pair['evidence'],pair['placebo'];r={'group':g,'keep':k,'evidence_chunks':list(map(int,e['job']['chunk_keeps'])),'placebo_chunks':list(map(int,p['job']['chunk_keeps'])),'evidence_correct':e['probes']['task']['correct'],'placebo_correct':p['probes']['task']['correct'],'evidence_answer':e['probes']['task']['answer'],'placebo_answer':p['probes']['task']['answer'],'evidence_margin':e['likelihood']['log_likelihood_margin'],'placebo_margin':p['likelihood']['log_likelihood_margin']};rows.append(r)
   lines.append(f"| {g} | {k} | {r['evidence_chunks']} | {r['evidence_correct']} | {r['placebo_chunks']} | {r['placebo_correct']} | {r['evidence_margin']:.4f} / {r['placebo_margin']:.4f} |")
  cases.append({'sample_id':sid,'role':item['role'],'full_correct':full['probes']['task']['correct'],'compressed_correct':base['probes']['task']['correct'],'full_margin':full['likelihood']['log_likelihood_margin'],'compressed_margin':base['likelihood']['log_likelihood_margin'],'comparisons':rows});lines.append('')
 lines += ['## 解释边界','- 同一案例的多个干预、重复chunk集合不是独立样本。','- keep32/64 同时改变表示、数量与位置锚点，不能单独据此归因 RoPE 或均值混合。','- placebo 使用同数量非证据chunk；恢复若非证据chunk也有效，应考虑全局注意力/数值与上下文干扰。','- 证据句包含标识符和属性值；没有只保护答案 token。路由仍使用已知证据，是 oracle 诊断。','- 未恢复不证明信息彻底丢失；恢复不证明通用无损。','- 全部耗时带诊断开销，不作为 prefill 加速结果。']
 summary={'audit':'PASS','completed_units':len(units),'expected_units':len(queue),'cases':cases,'limits':'Post-hoc answer-aware oracle diagnostic; no independent accuracy or clean speed claim.'}
 (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');(out/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'audit':'PASS','completed_units':len(units)}));return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();report(a.run)
