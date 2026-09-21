"""Finite sequential two-model run; no polling daemon and no GPU task killing."""
import argparse,json,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def save(p,x):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,indent=2)+'\n');temp.replace(p)
def main():
    a=argparse.ArgumentParser();a.add_argument('--out',required=True);args=a.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    source=out/'frozen_source';source.mkdir()
    for p in ROOT.iterdir():
        if p.is_file() and p.suffix in ('.py','.md','.jsonl'):shutil.copy2(p,source/p.name)
    statuses={};save(out/'pipeline_status.json',{'status':'RUNNING','models':statuses})
    try:
        with (out/'tests.log').open('w') as log:
            subprocess.run([sys.executable,str(source/'test_engine.py')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
        for name,path in [('qwen4b','<MODEL_ROOT>/Qwen3-4B-Instruct-2507'),('qwen06b','<MODEL_ROOT>/Qwen3-0.6B')]:
            if not (Path(path)/'model.safetensors').exists() and not (Path(path)/'model.safetensors.index.json').exists():
                statuses[name]={'status':'BLOCKED','reason':'model files unavailable'};continue
            statuses[name]={'status':'RUNNING'};save(out/'pipeline_status.json',{'status':'RUNNING','models':statuses})
            with (out/(name+'.log')).open('w') as log:
                result=subprocess.run([sys.executable,'-u',str(source/'run.py'),'--model',path,'--dataset',str(source/'dataset.jsonl'),'--out',str(out/name)],stdout=log,stderr=subprocess.STDOUT,timeout=12*3600)
            statuses[name]={'status':'COMPLETED' if result.returncode==0 else 'FAILED','returncode':result.returncode}
            save(out/'pipeline_status.json',{'status':'RUNNING','models':statuses})
            if result.returncode:raise RuntimeError(name+' failed; inspect model status/log')
        save(out/'pipeline_status.json',{'status':'COMPLETED' if all(v['status']=='COMPLETED' for v in statuses.values()) else 'PARTIAL','models':statuses})
    except BaseException as e:
        save(out/'pipeline_status.json',{'status':'FAILED','models':statuses,'error':repr(e),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
