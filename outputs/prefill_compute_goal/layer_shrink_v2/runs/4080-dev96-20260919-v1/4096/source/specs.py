"""Frozen query-blind conditions for the depth/budget development pilot."""
DATASET_SHA256='c775d3fcaf2cc3a6f32b3c06d8b2ccec02e800d9146f4b8ad3c329c2f8e8c3c5'
SPECS={
    'native_full': {'method':'identity','split_depth':12,'keep':64,'native':True},
    'identity_d12': {'method':'identity','split_depth':12,'keep':64,'native':False},
    'identity_d24': {'method':'identity','split_depth':24,'keep':64,'native':False},
    'mean_d12_k32': {'method':'pair_mean','split_depth':12,'keep':32,'native':False},
    'mean_d24_k32': {'method':'pair_mean','split_depth':24,'keep':32,'native':False},
    'mean_d12_k48': {'method':'pair_mean_48','split_depth':12,'keep':48,'native':False},
}
METHODS=tuple(SPECS)
IDENTITIES=('identity_d12','identity_d24')
CANDIDATES=('mean_d12_k32','mean_d24_k32','mean_d12_k48')
TIMING_METHODS=('native_b1024','native_b4096',*IDENTITIES,*CANDIDATES)

def expected_shape(lengths,method,layers=36):
    spec=SPECS[method];full=sum(lengths.values())
    if lengths['T']%64:raise ValueError('whole T chunks required')
    upper=full-lengths['T']+lengths['T']//64*spec['keep']
    depth=spec['split_depth']
    return full,upper,(depth*full+(layers-depth)*upper)/(layers*full)

LENGTHS=(4096,8192,16384)
TIMING_INSTANCE=0
