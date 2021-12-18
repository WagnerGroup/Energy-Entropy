import snakemake
import numpy as np
import itertools

snakemake.snakemake("Snakefile",cores=1,
                    targets=['h2_1.4/hf/vtz/vmc_hci0.01_0.005_20.chk', 
                            'h2_1.4/hf/vtz/dmc_hci0.01_0.005_20_0.02.chk']
                    )#,forcetargets=True

