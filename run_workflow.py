# import snakemake
import numpy as np
import itertools
import os
import os.path
import set_geom
from set_geom import make_paths
import glob


molecules = ['h2','h4','h6'] #,'h8','h10'
bond_lengths = [1.4,2.0,3.0] #1.0,,4.0
basis = ['vtz'] #'vdz',,'vqz',,'v5z'

folder = itertools.product(molecules,bond_lengths)
for m,length in folder:
    make_paths([m,length])

prod = itertools.product(molecules,basis,bond_lengths)

### start from HCI calculations

hf_target, cc_target, hci_target, fci_target = [],[],[],[]
tolerances = {"h2": [0.1,0.08,0.05,0.02],
                "h4": [0.1,0.08,0.05,0.02,0.01,0.008,0.005],
                "h6":[0.1,0.08,0.05,0.02,0.01,0.008,0.005,0.002,0.001]} #

for molecule, b, length in prod:
    dir_hf = f"{molecule}_{length}/hf/{b}"
    hf_target.append(f"{dir_hf}/mf.chk")
    cc_target.append(f"{dir_hf}/cc.chk")

    for tol in tolerances[molecule]:
        hci_target.append(f"{dir_hf}/hci{tol}.chk")

    if molecule == 'h6':
        pass
    else:
        fci_target.append(f"{dir_hf}/fci.chk")

targets =  hf_target + hci_target + cc_target + fci_target
files = " ".join(targets)
print("Total number of tasks:",len(targets))
print(files)

f = open(f"targets.txt","w")
f.write(files)
f.write("\n")


### select best HCI wavefunctions
hci_wf = {"h2": 0.01, "h4": 0.005, "h6": 0.001}

### start qmc calculations

vmc_target, dmc_target = [],[]
tstep = 0.02
determinant_cutoff = '0.000001'
orbitals = 'large'
statenumber = 0

nconfigs = [3200,400,800,1600,6400,12800] #,25600
dmc_nconfig =nconfigs[-1]

prod = itertools.product(molecules,basis,bond_lengths)
for molecule, b, length in prod:
    dir_hf = f"{molecule}_{length}/hf/{b}"
    dmc_target.append(f"{dir_hf}/dmc_mf_{orbitals}_{statenumber}_{dmc_nconfig}_{tstep}.chk") 
    hci_tol = hci_wf[molecule]
    dmc_target.append(f"{dir_hf}/dmc_hci{hci_tol}_{determinant_cutoff}_{orbitals}_{statenumber}_{dmc_nconfig}_{tstep}.chk")
    for nconfig in nconfigs:
        vmc_target.append(f"{dir_hf}/vmc_mf_{orbitals}_{statenumber}_{nconfig}.chk") 
        vmc_target.append(f"{dir_hf}/vmc_hci{hci_tol}_{determinant_cutoff}_{orbitals}_{statenumber}_{nconfig}.chk")
        

targets =  vmc_target + dmc_target
files = " ".join(targets)
print("Total number of tasks:",len(targets))
print(files)


f.write("\n")
f.write(files)
f.write("\n")
f.close
