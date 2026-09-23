"""CPU-only evidence localization using the exact model tokenizer."""
import argparse,hashlib,json,re
from pathlib import Path
from transformers import AutoTokenizer
from data import encode,expected_from_text
ROOT=Path(__file__).resolve().parent

def locate(tok,item):
 row=item['row']; assert row['task']=='narrative' and row['gold']==expected_from_text(row)
 ex=encode(tok,row); enc=tok(row['T'],add_special_tokens=False,return_offsets_mapping=True)
 assert enc['input_ids']==ex['segments']['T']; offsets=enc['offset_mapping']; s_len=len(ex['segments']['S'])
 def span(a,b):
  ii=[i for i,(x,y) in enumerate(offsets) if y>x and y>a and x<b]; assert ii
  return {'char_start':a,'char_end':b,'text':row['T'][a:b],'token_start':ii[0],'token_end_exclusive':ii[-1]+1,'chunks':sorted({i//64 for i in ii}),'tokens':[{'t_index':i,'chunk':i//64,'offset_in_chunk':i%64,'group16':(i%64)//4,'offset_in_group16':i%4,'group16_anchor_logical_position':s_len+(i//4)*4+3,'decoded':tok.decode([enc['input_ids'][i]])} for i in ii]}
 patterns={'warehouse':r'The parcel labelled '+re.escape(row['key'])+r' was delivered to the (\w+) warehouse\.','boxes':r'The parcel labelled '+re.escape(row['key'])+r' contains (\d+) sealed boxes\.'}
 evidence={}
 for field,pat in patterns.items():
  ms=list(re.finditer(pat,row['T']));assert len(ms)==1
  m=ms[0]; evidence[field]={'sentence':span(*m.span()),'value':span(*m.span(1))}
 all_chunks=sorted(set(c for e in evidence.values() for c in e['sentence']['chunks']))
 groups={f'chunk_{c}':[c] for c in all_chunks}
 groups['warehouse_sentence']=evidence['warehouse']['sentence']['chunks']
 groups['all_evidence']=all_chunks
 placebo={}
 available=[i for i in range(row['target_tokens']//64) if i not in all_chunks]
 for name,cs in groups.items():
  # Nearest preceding non-evidence chunks; fall back forward only if none before.
  rank=sorted(available,key=lambda c:(c>=min(cs),abs(c-min(cs)),c))
  placebo[name]=sorted(rank[:len(cs)])
 assert all(len(placebo[n])==len(cs) and not set(placebo[n])&set(all_chunks) for n,cs in groups.items())
 return {'sample_id':row['sample_id'],'role':item['role'],'S_tokens':s_len,'T_tokens':len(enc['input_ids']),'evidence':evidence,'intervention_groups':groups,'placebo_groups':placebo}

def make_queue(locations):
 jobs=[]
 for loc in locations:
  sid=loc['sample_id']
  for method in ('full','compressed'):
   jobs.append({'sample_id':sid,'method':method,'chunk_keeps':{},'unit_key':sid+'__'+method})
  for group,chunks in loc['intervention_groups'].items():
   for kind,cs in [('evidence',chunks),('placebo',loc['placebo_groups'][group])]:
    for keep in (32,64):
     jobs.append({'sample_id':sid,'method':'intervention','group':group,'kind':kind,'chunk_keeps':{str(c):keep for c in cs},'unit_key':f'{sid}__{group}__{kind}__keep{keep}'})
 assert len({j['unit_key'] for j in jobs})==len(jobs)
 return jobs

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);a=p.parse_args();tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);assert tok.is_fast
 selected=json.loads((ROOT/'selection.json').read_text());locations=[locate(tok,x) for x in selected['samples']];q=make_queue(locations)
 obj={'status':'READY_GPU_NOT_RUN','selection_sha256':hashlib.sha256((ROOT/'selection.json').read_bytes()).hexdigest(),'tokenizer_sha256':hashlib.sha256((Path(a.model)/'tokenizer.json').read_bytes()).hexdigest(),'block':1024,'depth':30,'base_keep':16,'index_convention':'zero-based chunks and tokens relative to T; logical RoPE positions include S; end offsets exclusive','locations':locations,'queue':q,'expected_units':len(q)}
 (ROOT/'prepared.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
 lines=['# v5 失败案例：GPU 干预前的离线诊断','', '状态：证据定位已完成，GPU 干预尚未运行；不能声称已恢复答案。','', '全部三个失败是 narrative/16K，仓库 Amber→Dune，箱子数保持正确。','', '| 样本 | 角色 | 正确仓库/箱数 | 原压缩输出 | 仓库证据 chunk | 箱数证据 chunk |','|---|---|---|---|---|---|']
 for item,loc in zip(selected['samples'],locations):
  r=item['row'];ans=item['historical']['d30_mean16_b1024']['probes']['task']['answer'].replace('\n',' ')
  lines.append(f"| {r['sample_id']} | {item['role']} | {r['gold']} | {ans} | {loc['evidence']['warehouse']['sentence']['chunks']} | {loc['evidence']['boxes']['sentence']['chunks']} |")
 lines+=['','索引从0开始，chunk长64 token。prepared.json 记录每个证据 token、合并组、组内偏移和原逻辑位置锚点。','',f'固定队列：{len(q)} 单元。每份文档重新验证 Full identity gate 和压缩基线；证据 chunk 单独及联合改为 keep32/64，并设置同数量、相同预算的非证据 placebo chunk。','', '这是使用题目和答案定位证据的事后机制诊断，不是 query-blind 路由算法，也不能用于宣称泛化准确率。正确对照只存在2份同为 Amber 的样本，第3份匹配任务/长度但仓库为 Dune；详见 selection.json。','', '一次恢复只能证明该配置的干预影响输出，不能单独分离内容混合、位置锚定、全局注意力归一化等机制。未恢复也不等于证据已经不可逆丢失。']
 (ROOT/'OFFLINE_REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'status':'READY_GPU_NOT_RUN','documents':len(locations),'expected_units':len(q)}))
if __name__=='__main__':main()
