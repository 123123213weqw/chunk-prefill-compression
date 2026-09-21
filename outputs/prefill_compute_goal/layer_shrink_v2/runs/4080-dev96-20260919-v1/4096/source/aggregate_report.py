#!/usr/bin/env python3
"""Aggregate only three fully audited, source-identical length stages."""
import argparse,json
from pathlib import Path
from report_experiment import audit,CANDIDATES,EXPECTED
from paired_stats import comparison,strata_scores

def aggregate(root):
    root=Path(root);paths=[root/str(n) for n in (4096,8192,16384)]
    reports=[audit(p) for p in paths]
    manifests=[json.loads((p/'manifest.json').read_text()) for p in paths]
    for m in manifests[1:]:
        for key in ('source_sha256','dataset_sha256','model_config_sha256','torch','transformers','dtype','gpu'):
            assert m[key]==manifests[0][key],('cross-stage drift',key)
    rows=[r for p in paths for r in json.loads((p/'correctness.json').read_text())]
    assert len(rows)==1152 and len({r['sample_id'] for r in rows})==96 and len({r['pair_id'] for r in rows})==48
    scores={}
    for m in EXPECTED:
        rr=[r for r in rows if r['method']==m and r['probe']=='readout'];cc=[r for r in rows if r['method']==m and r['probe']=='control']
        scores[m]={'n':96,'readout_correct':sum(r['correct'] for r in rr),'control_correct':sum(r['correct'] for r in cc),
                   'field_correct':sum(r['field_correct'] for r in rr),'fields':sum(r['fields'] for r in rr),
                   'unavailable_readout':sum(r['status']!='OK' for r in rr),'unavailable_control':sum(r['status']!='OK' for r in cc)}
    result={'status':'PASS','samples':96,'pairs':48,'test_used':False,'scores':scores,
            'source_sha256':manifests[0]['source_sha256'],'stages':[{'length':a['length'],'audit':'PASS'} for a in reports],
            'paired_comparisons':{m:comparison(rows,m) for m in CANDIDATES},'stratified_scores':strata_scores(rows),
            'all_trial_boundaries_isolated':all(a['all_trial_boundaries_isolated'] for a in reports)}
    lines=['# 完整开发集 96 条 / 48 对：结果','',
           '仅覆盖两种合成模板，不是新数据盲测；反事实成员不拆开统计。',
           '', '| 方法 | readout | control | 字段命中 | 读取 OOM |','|---|---:|---:|---:|---:|']
    for m,v in scores.items():lines.append(f"| {m} | {v['readout_correct']}/96 | {v['control_correct']}/96 | {v['field_correct']}/{v['fields']} | {v['unavailable_readout']} |")
    lines+=['','## 相对 Full 损伤','','| 方法 | Full 对→方法错 | Full 错→方法对 | 准确率差 (百分点) | 95% CI (百分点) |','|---|---:|---:|---:|---|']
    for m,c in result['paired_comparisons'].items():
        lo,hi=c['ci95'];lines.append(f"| {m} | {c['full_correct_to_method_wrong']} | {c['full_wrong_to_method_correct']} | {100*c['method_minus_full']:.2f} | [{100*lo:.2f}, {100*hi:.2f}] |")
    lines+=['2000 次 template×length×layout 分层 pair-cluster bootstrap；描述性开发统计，不作正式无损/非劣证明。',
            '', '## 按长度', '', '| 方法 | T 长度 | readout | control |','|---|---:|---:|---:|']
    for a in reports:
        for m,v in a['scores'].items():lines.append(f"| {m} | {a['length']} | {v['readout_correct']}/32 | {v['control_correct']}/32 |")
    lines+=['','## 按长度、模板、布局细分','','| 方法 | 长度 | 模板 | 布局 | 正确 | OOM |','|---|---:|---|---|---:|---:|']
    for r in result['stratified_scores']:
        if r['probe']=='readout' and r['method'] not in ('identity_d12','identity_d24'):
            lines.append(f"| {r['method']} | {r['length']} | {r['template']} | {r['layout']} | {r['correct']}/{r['n']} | {r['unavailable']} |")
    lines+=['','## 性能与边界','',
            '性能仅覆盖预先固定 12 个文档，每方法 1 warmup + 5 repeats；不等于 96 文档的性能分布。',
            '全部 trial 边界未观察到其他 compute PID；仍不能排除边界之间的干扰。' if result['all_trial_boundaries_isolated'] else '**有 trial 非隔离或遥测未知，不以污染数据作可靠提速结论。**',
            '不根据分数筛掉 Full 失败、压缩失败或 OOM。审计 PASS 只说明数据一致。',
            '', '完整形状、逐条答案、计时和环境见三个阶段目录：']
    for n in (4096,8192,16384):lines.append(f'- [{n} 阶段报告]({n}/REPORT_ZH.md)')
    (root/'AGGREGATE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (root/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();r=aggregate(a.directory)
    print(json.dumps({'status':r['status'],'samples':96,'scores':r['scores']},ensure_ascii=False))
if __name__=='__main__':main()
