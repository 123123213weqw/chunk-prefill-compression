"""Re-score and cross-check raw inverse-intervention records without GPU."""
import argparse,hashlib,json,math
from pathlib import Path
from scoring import strict_score
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def report(out):
 manifest=json.loads((out/'manifest.json').read_text());prep=json.loads((ROOT/'queue_v6.json').read_text());selection=json.loads((ROOT/'selection.json').read_text());parent=json.loads((ROOT/'PARENT_DIAGNOSIS_SUMMARY.json').read_text());jobs=prep['queue'];assert len(jobs)==36
 assert json.loads((out/'queue.json').read_text())==jobs
 assert prep['selection_sha256']==sha(ROOT/'selection.json') and prep['parent_prepared_sha256']==sha(ROOT/'prepared.json')
 for name,value in manifest['source_sha256'].items():assert sha(ROOT/name)==value,name
 files=list((out/'units').glob('*.json'));assert len(files)==len(jobs)==manifest['expected_units'];units={p.stem:json.loads(p.read_text()) for p in files};samples={x['row']['sample_id']:x for x in selection['samples']}
 for j in jobs:
  u=units[j['unit_key']];assert u['status']=='COMPLETED' and u['job']==j and u['shape']['status']=='PASS'
  row=samples[j['sample_id']]['row']
  for name,p in u['probes'].items():
   gold=row['gold'] if name=='task' else {'label':'ready-blue'};assert p['gold']==gold
   for k,v in strict_score(p['answer'],gold).items():assert p[k]==v
  assert u['probes']['control']['correct'] and math.isfinite(u['likelihood']['log_likelihood_margin'])
  if j['method'] in ('full','all_compressed'):
   assert u['historical_reproduction']=='PASS';old=samples[j['sample_id']]['historical'][('full' if j['method']=='full' else 'd30_mean16')+'_b1024']
   for name in ('task','control'):assert u['probes'][name]['ids']==old['probes'][name]['ids']
   if j['method']=='all_compressed':assert u['legacy_engine_equivalence']=='PASS'
   else:
    for g in u['gates']:assert g['ids_equal'] and g['kv_rms']<=.005 and g['logit_rms']<=.005 and g['kl']<=.001
  else:
   keep=[j['chunk_keeps'].get(str(i),64) for i in range(row['target_tokens']//64)];assert [r['keep'] for r in u['routes']]==keep
   assert u['shape']['saved_T_layer_tokens']==sum(6*(64-k) for k in keep)
 for item in selection['samples']:
  sid=item['row']['sample_id']
  for g in ('warehouse_sentence','all_evidence'):
   a,b=(units[f'{sid}__{g}__{kind}__compress16'] for kind in ('evidence','placebo'))
   assert a['shape']['saved_T_layer_tokens']==b['shape']['saved_T_layer_tokens']
 lines=['# v6 反向局部压缩诊断','',f'审计 PASS；完成 {len(units)}/36 单元。六份文档来自已暴露v5，不能作独立泛化准确率。','', '与先前“压缩所有chunk、只保护指定chunk”相反，本次“保留所有chunk、只压缩指定chunk”。每个证据干预有相同压缩预算的非证据对照。','', '| 样本 | 角色 | Full | 全压缩 | 只压仓库证据 | 只压非证据对照 | 只压两处证据 | 只压两个非证据对照 |','|---|---|---|---|---|---|---|---|'];cases=[]
 prev={x['sample_id']:x for x in parent['cases']}
 for item in selection['samples']:
  sid=item['row']['sample_id'];u=lambda key:units[f'{sid}__{key}'];vals={'full':u('full')['probes']['task']['correct'],'all_compressed':u('all_compressed')['probes']['task']['correct']}
  for g in ('warehouse_sentence','all_evidence'):
   for kind in ('evidence','placebo'):
    x=u(f'{g}__{kind}__compress16');vals[f'{g}_{kind}']={'correct':x['probes']['task']['correct'],'answer':x['probes']['task']['answer'],'margin':x['likelihood']['log_likelihood_margin'],'compressed_chunks':list(map(int,x['job']['chunk_keeps'])),'saved_T_layer_tokens':x['shape']['saved_T_layer_tokens']}
  old=prev[sid];vals['prior_protect_warehouse64']=[c for c in old['comparisons'] if c['group']=='warehouse_sentence' and c['keep']==64][0]['evidence_correct'];vals['prior_protect_all64']=[c for c in old['comparisons'] if c['group']=='all_evidence' and c['keep']==64][0]['evidence_correct'];cases.append({'sample_id':sid,'role':item['role'],**vals})
  lines.append('| '+' | '.join([sid,item['role'],str(vals['full']),str(vals['all_compressed']),str(vals['warehouse_sentence_evidence']['correct']),str(vals['warehouse_sentence_placebo']['correct']),str(vals['all_evidence_evidence']['correct']),str(vals['all_evidence_placebo']['correct'])])+' |')
 lines+=['','## 与先前保护实验并读','','| 失败样本 | 全压缩 | 全压缩但保护仓库证据 | 仅压仓库证据 | 全压缩但保护两处证据 | 仅压两处证据 |','|---|---|---|---|---|---|']
 for x in cases:
  if x['role']=='loss':lines.append('| '+' | '.join(str(x[k] if not isinstance(x[k],dict) else x[k]['correct']) for k in ('sample_id','all_compressed','prior_protect_warehouse64','warehouse_sentence_evidence','prior_protect_all64','all_evidence_evidence'))+' |')
 lines+=['','## 限制','', '- 本轮干预使用答案定位；每个干预与对照预算相等，但不同基线（全保留/全压缩）不能直接比较绝对压缩率。','- 只压缩某个chunk后模型出错，表明在该上下文中该局部变化足以改变输出；不能单独归因于信息内容、RoPE位置、注意力权重归一化或数值误差。','- 只压证据和只压非证据若都正确，不能推断原全局压缩的错误来自某一单个chunk；可能是交互。','- 多个条件共享相同文档，不是独立样本。logP对照使用紧凑JSON teacher forcing，不是校准置信度；有诊断开销的时间不是加速数字。']
 summary={'audit':'PASS','expected_units':36,'completed_units':len(units),'cases':cases,'notes':'Post-hoc inverse mechanism diagnostic; no independent accuracy or clean timing claim.'};(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');(out/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'audit':'PASS','completed_units':len(units)}));return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();report(a.run)
