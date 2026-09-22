import argparse,json,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def save(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--model',required=True);a=p.parse_args();out=a.out;out.mkdir(parents=True,exist_ok=False);source=out/'frozen_source';source.mkdir()
 for f in ROOT.iterdir():
  if f.is_file() and f.suffix in ('.py','.md','.json','.jsonl'):shutil.copy2(f,source/f.name)
 save(out/'supervisor_status.json',{'status':'TESTING'})
 try:
  with (out/'tests.log').open('w') as log:
   for name in ('test_engine.py','test_long.py','test_scores.py','test_diagnostic.py'):
    subprocess.run([sys.executable,str(source/name)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
  save(out/'supervisor_status.json',{'status':'RUNNING'})
  with (out/'worker.log').open('w') as log:r=subprocess.run([sys.executable,'-u',str(source/'run.py'),'--out',str(out/'results'),'--model',a.model],stdout=log,stderr=subprocess.STDOUT,timeout=8100)
  status=json.loads((out/'results/status.json').read_text());save(out/'supervisor_status.json',{'status':status['status'],'returncode':r.returncode,'worker':status})
 except BaseException as e:
  save(out/'supervisor_status.json',{'status':'FAILED','error':repr(e),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
