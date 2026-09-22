"""Descriptive full-v5 narrative breakdown; does not select additional interventions."""
import argparse,json
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--source-run',type=Path,required=True);a=p.parse_args();rows=[json.loads(x) for x in (a.source_run/'frozen_source/fresh.jsonl').read_text().splitlines()];cells=defaultdict(lambda:{'n':0,'full_correct':0,'compressed_correct':0,'lost':0,'gained':0})
 for r in rows:
  if r['task']!='narrative':continue
  u={m:json.loads((a.source_run/f'results/units/correctness__{r["sample_id"]}__{m}__b1024__rnone.json').read_text()) for m in ('full','d30_mean16')};f,c=[u[m]['probes']['task']['correct'] for m in ('full','d30_mean16')];d=cells[(r['target_tokens'],r['gold']['warehouse'])];d['n']+=1;d['full_correct']+=f;d['compressed_correct']+=c;d['lost']+=f and not c;d['gained']+=c and not f
 prepared=json.loads((ROOT/'prepared.json').read_text());phase=[]
 for loc in prepared['locations']:
  phase.append({'sample_id':loc['sample_id'],'role':loc['role'],'warehouse_tokens':loc['evidence']['warehouse']['value']['tokens']})
 obj={'scope':'Descriptive all-v5 narrative b1024 breakdown; post-hoc, no significance or independence claim.','cells':[{'target_tokens':k[0],'warehouse':k[1],**v} for k,v in sorted(cells.items())],'selected_warehouse_group_offsets':phase}
 (ROOT/'HISTORY_ANALYSIS.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
 lines=['# 原始 v5 的错误集中性检查','','统一使用 block1024；按 gold 仓库分组，而不是模型输出分组。','', '| 长度 | 仓库 | 样本数 | Full 正确 | 压缩正确 | 丢失 | 新增正确 |','|---|---|---:|---:|---:|---:|---:|']
 for r in obj['cells']:lines.append('| '+' | '.join(str(r[k]) for k in ('target_tokens','warehouse','n','full_correct','compressed_correct','lost','gained'))+' |')
 lines+=['','## 证据在合并组中的位置','','3个失败样本的 Amber token 在4-token均值组内的偏移分别为0、1、3；两个正确 Amber 对照分别为3、1。因此“只要不是组末就丢失”不符合这些记录。这里只是排除一个过强的描述性规则，不证明位置与错误无关。','','这些数据同时存在 Full 自身的仓库错误：不能把所有实体检索错误都归因压缩。恢复实验需要先复现原输出，再对比同预算非证据chunk。','', '本文件不包含新的GPU干预结果。']
 (ROOT/'HISTORY_ANALYSIS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(obj['cells']))
if __name__=='__main__':main()
