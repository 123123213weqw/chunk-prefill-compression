METHODS={'d27_mean32':{'kind':'fixed','depth':27,'keep':32},
         'd27_endpoint32':{'kind':'fixed','depth':27,'keep':32,'merge':'endpoint'},
         'd30_mean16':{'kind':'fixed','depth':30,'keep':16}}
BLOCKS=(1024,4096)
SEEDS=(20260926,20260927,20260928,20260929)
TIMING_SEEDS=SEEDS[:2]
WARMUPS=2
REPEATS=7
