"""Full raw-record audit and non-monotone count-trajectory report; CPU only."""
import argparse,hashlib,json
from pathlib import Path
from scoring import strict_score
from prepare_v7 import make,COUNTS,ORDERS
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def report(out):
 manifest=json.loads((out/'manifest.json').read_text());prep=json.loads((ROOT/'queue_v7.json').read_text());selected=json.loads((ROOT/'selection.json').read_text());planned=make()
 assert prep['queue']==planned['queue'] and prep['orders']==planned['orders'] and prep['expected_units']==228
 assert prep['selection_sha256']==sha(ROOT/'selection.json') and prep['parent_prepared_sha256']==sha(ROOT/'prepared.json') and prep['parent_v6_summary_sha256']==sha(ROOT/'PARENT_V6_SUMMARY.json')
 assert json.loads((out/'queue.json').read_text())==prep['queue']
 for name,value in manifest['source_sha256'].items():assert sha(ROOT/name)==value,name
 files=list((out/'units').glob('*.json'));assert len(files)==len(prep['queue'])==manifest['expected_units']==228;units={p.stem:json.loads(p.read_text()) for p in files};samples={x['row']['sample_id']:x for x in selected['samples']}
 for j in prep['queue']:
  u=units[j['unit_key']];assert u['status']=='COMPLETED' and u['job']==j and u['shape']['status']=='PASS'
  row=samples[j['sample_id']]['row'];assert u['sample_id']==j['sample_id']
  for name,p in u['probes'].items():
   gold=row['gold'] if name=='task' else {'label':'ready-blue'};assert p['gold']==gold
   for k,v in strict_score(p['answer'],gold).items():assert p[k]==v
  assert u['probes']['control']['correct']
  if j['method'] in ('full','all_compressed'):
   assert u['historical_reproduction']=='PASS';old=samples[j['sample_id']]['historical'][('full' if j['method']=='full' else 'd30_mean16')+'_b1024']
   for name in ('task','control'):assert u['probes'][name]['ids']==old['probes'][name]['ids']
   if j['method']=='all_compressed':assert u['legacy_engine_equivalence']=='PASS'
   else:
    for g in u['gates']:assert g['ids_equal'] and g['kv_rms']<=.005 and g['logit_rms']<=.005 and g['kl']<=.001
  else:
   assert j['chunk_indices']==prep['orders'][j['sample_id']][j['order']][:j['count']]
   subset=set(j['chunk_indices']);assert len(subset)==j['count'];assert [r['keep'] for r in u['routes']]==[16 if i in subset else 64 for i in range(256)]
   assert u['shape']['compressed_chunks']==j['count'] and u['shape']['saved_T_layer_tokens']==6*48*j['count']
 locations={x['sample_id']:x for x in json.loads((ROOT/'prepared.json').read_text())['locations']};trajectories=[]
 for item in selected['samples']:
  sid=item['row']['sample_id'];assert units[sid+'__full']['probes']['task']['correct'];loc=locations[sid]
  warehouse=set(loc['evidence']['warehouse']['sentence']['chunks']);boxes=set(loc['evidence']['boxes']['sentence']['chunks'])
  for order in ORDERS:
   points=[]
   for count in COUNTS:
    key=sid+('___' if False else '__')+('full' if count==0 else 'all_compressed' if count==256 else f'{order}__n{count}')
    u=units[key];subset=set() if count==0 else set(range(256)) if count==256 else set(prep['orders'][sid][order][:count]);points.append({'count':count,'correct':u['probes']['task']['correct'],'answer':u['probes']['task']['answer'],'generated_ids':u['probes']['task']['ids'],'warehouse_chunk_compressed':warehouse<=subset,'box_chunk_compressed':boxes<=subset})
   transitions=[{'from_count':a['count'],'to_count':b['count'],'from_correct':a['correct'],'to_correct':b['correct']} for a,b in zip(points,points[1:]) if a['correct']!=b['correct']]
   trajectories.append({'sample_id':sid,'role':item['role'],'order':order,'points':points,'first_failure_count':next((p['count'] for p in points if not p['correct']),None),'transitions':transitions,'non_monotone_recovery':any(not x['from_correct'] and x['to_correct'] for x in transitions)})
 lines=['# v7 chunk数量扫描','', '审计PASS；228/228原子单元，6份已暴露文档×4条固定压缩顺序；0与256端点共用，不重复计为独立GPU试验。', '', 'count是256个T chunk中压缩到16个token的数量。每层投影/MLP相关T token节省 = count×48×6；不是总计算或实际加速。','', '| 样本 | 原角色 | 顺序 | 0/16/32/64/96/128/160/192/224/240/256 的正确性（✓/×） | 首错grid点 | 翻转次数 | 错后恢复 |','|---|---|---|---|---:|---:|---|']
 for t in trajectories:
  marks=''.join('✓' if p['correct'] else '×' for p in t['points']);lines.append(f"| {t['sample_id']} | {t['role']} | {t['order']} | {marks} | {t['first_failure_count']} | {len(t['transitions'])} | {t['non_monotone_recovery']} |")
 lines+=['','## 解读边界','','- 图格代表预先选定的离散压缩数量，不是连续真实阈值；同一顺序下新增chunk是嵌套的。','- 如果出现错后恢复，单调阈值假设不成立；所有翻转逐项记录在summary.json。','- Prefix/suffix/random是chunk数量与位置的机制对照，并非模型可用的选择器；每份文档的4条轨迹不是4份独立样本。','- 只对3个原失败案例和3个正确对照做事后扫描，不可用于独立准确率或泛化宣称。','- 计时含hooks及评分，不能代替干净prefill测速。']
 summary={'audit':'PASS','expected_units':228,'completed_units':len(units),'counts':COUNTS,'orders':ORDERS,'trajectories':trajectories,'notes':'Answer-blind route ordering on already-exposed cases; descriptive mechanism scan only.'};(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');(out/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'audit':'PASS','completed_units':len(units),'trajectories':len(trajectories)}));return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();report(a.run)
