import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
 parent=json.loads((ROOT/'prepared.json').read_text());selected=json.loads((ROOT/'selection.json').read_text())
 assert parent['selection_sha256']==hashlib.sha256((ROOT/'selection.json').read_bytes()).hexdigest()
 jobs=[]
 for item in selected['samples']:
  sid=item['row']['sample_id'];loc=next(x for x in parent['locations'] if x['sample_id']==sid)
  for method in ('full','all_compressed'):jobs.append({'sample_id':sid,'method':method,'chunk_keeps':{},'unit_key':sid+'__'+method})
  for group in ('warehouse_sentence','all_evidence'):
   for kind,cs in (('evidence',loc['intervention_groups'][group]),('placebo',loc['placebo_groups'][group])):
    jobs.append({'sample_id':sid,'method':'local_only','group':group,'kind':kind,'chunk_keeps':{str(c):16 for c in cs},'unit_key':f'{sid}__{group}__{kind}__compress16'})
 assert len(jobs)==36 and len({j['unit_key'] for j in jobs})==len(jobs)
 out={'status':'READY_GPU_NOT_RUN','parent_prepared_sha256':hashlib.sha256((ROOT/'prepared.json').read_bytes()).hexdigest(),'selection_sha256':hashlib.sha256((ROOT/'selection.json').read_bytes()).hexdigest(),'block':1024,'depth':30,'default_keep':64,'override_keep':16,'expected_units':len(jobs),'queue':jobs}
 (ROOT/'queue_v6.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'expected_units':len(jobs),'post_hoc':True}))
if __name__=='__main__':main()
