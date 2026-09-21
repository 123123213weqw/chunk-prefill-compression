#!/usr/bin/env python3
"""Finite, detached-capable three-stage supervisor. Never modifies live sources."""
import argparse,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--dataset',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    files=[*sorted(ROOT.glob('*.py')),ROOT/'PROTOCOL.md',ROOT.parents[1]/'validation_v3/methods.py',Path(a.dataset).resolve()]
    hashes={str(p):digest(p) for p in files}
    state={'status':'running','pid':os.getpid(),'started_unix':time.time(),'stages':{},'input_sha256':hashes}
    dump(out/'PIPELINE.json',state)
    def execute(cmd,log,env=None):
        with log.open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
    try:
        for length in (4096,8192,16384):
            assert {str(p):digest(p) for p in files}==hashes,'source or dataset changed between stages'
            run=out/str(length);state['active_length']=length;state['stages'][str(length)]='running';dump(out/'PIPELINE.json',state)
            execute([sys.executable,'-u',str(ROOT/'run_experiment.py'),'--model',a.model,'--dataset',a.dataset,'--length',str(length),'--out',str(run)],out/f'{length}.log')
            execute([sys.executable,str(run/'source/report_experiment.py'),str(run)],out/f'{length}-audit.log')
            execute([sys.executable,str(run/'source/test_report.py')],out/f'{length}-tamper-tests.log',dict(os.environ,RUN_DIRECTORY=str(run)))
            state['stages'][str(length)]='completed_audited';dump(out/'PIPELINE.json',state)
        execute([sys.executable,str(ROOT/'aggregate_report.py'),str(out)],out/'aggregate.log')
        state.update(status='completed',finished_unix=time.time());dump(out/'PIPELINE.json',state)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),finished_unix=time.time());dump(out/'PIPELINE.json',state);raise
if __name__=='__main__':main()
