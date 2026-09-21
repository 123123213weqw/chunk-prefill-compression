def fixed(d,k,**kw):return {'kind':'fixed','depth':d,'keep':k,**kw}
METHODS={**{f'd{d}_k32':fixed(d,32) for d in (12,18,24,27,30)},'d18_k48':fixed(18,48),'d27_k48':fixed(27,48),
 'd24_k40':fixed(24,40),'d30_k16':fixed(30,16),'d18_k48_floor':fixed(18,48,phase='floor'),
 'd18_k48_endpoint':fixed(18,48,merge='endpoint'),'d27_k32_endpoint':fixed(27,32,merge='endpoint'),
 'adaptive_equal':{'kind':'adaptive_equal'},'random_equal':{'kind':'random_equal'}}
BASES={'d18_k48':METHODS['d18_k48'],'d27_k32':METHODS['d27_k32']}
