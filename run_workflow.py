# import snakemake
import numpy as np
import itertools
import os
import os.path
import set_geom
from set_geom import make_paths
import glob

"""
def generate_mc_hci(dir,nconfig,tol,orbitals,determinant_cutoff=0,tstep=0.02,statenumber=0):
    vmc_hci, dmc_hci=[], [], []
    vmc_nconfig = nconfig[0]
    dmc_nconfig = nconfig[1]
    mc_tol = tol
    
    prod =itertools.product(vmc_nconfig,mc_tol,orbitals)    
    for nconfig,tol,orbital in prod:    
        if nconfig>=3200 and orbital=='orbitals':
            orbital='large'
        vmc_hci.append(f"{dir}/vmc_hci{tol}_{determinant_cutoff}_{orbital}_{statenumber}_{nconfig}.chk")
    
    prod2 =itertools.product(dmc_nconfig,mc_tol,orbitals)    
    for nconfig,tol,orbital in prod2:    
        if nconfig>=3200 and orbital=='orbitals':
            orbital='large'
        dmc_hci.append(f"{dir}/dmc_hci{tol}_{determinant_cutoff}_{orbital}_{statenumber}_{nconfig}_{tstep}.chk")
    return vmc_hci, dmc_hci
"""

def generate_vmc_hf(dir,nconfig,orbitals):
    vmc_target=[]
    prod=itertools.product(nconfig,orbitals)
    for nconfig,orbital in prod:
        if nconfig>=3200 and orbitals=='orbitals':
            orbital='large'
        vmc_target.append(f"{dir}/vmc_mf_{orbital}_0_{nconfig}.chk") 
    return vmc_target

def generate_dmc_hf(dir,nconfig,orbitals,tstep=0.02):
    dmc_target=[]
    prod=itertools.product(nconfig,orbitals)
    for nconfig,orbital in prod:
        if nconfig>=3200 and orbital=='orbitals':
            orbital='large'
        dmc_target.append(f"{dir}/dmc_mf_{orbital}_0_{nconfig}_{tstep}.chk") 
    return dmc_target



molecule = ['h2','h4','h6']#,'h8','h10'
bond_lengths = [1.4,2.0,3.0] #1.0,,4.0
basis = ['vtz'] #'vdz',,'vqz',,'v5z'

folder = itertools.product(molecule,bond_lengths)
for molecule,length in folder:
    make_paths([molecule,length])
prod = itertools.product(molecule,basis,bond_lengths)


### start from HCI calculations

hf_target, cc_target, hci_target, fci_target = [],[],[],[],[]
tolerance = [0.1,0.08,0.05,0.02,0.01,0.008] #

for molecule,basis,length in prod:

    dir_hf = f"{molecule}_{length}/hf/{basis}"
    hf_target.append(f"{dir_hf}/mf.chk")
    cc_target.append(f"{dir_hf}/cc.chk")

    for tol in tolerance:
        hci_target.append(f"{dir_hf}/hci{tol}.chk")

    if molecule == 'h6':
        pass
    else:
        fci_target.append(f"{dir_hf}/fci.chk")

    
targets =  hf_target + hci_target + cc_target + fci_target
files = " ".join(targets)
print("Total number of tasks:",len(targets))

f = open(f"targets.txt","w")
f.write(files)
f.close


"""
vmc_target, dmc_target = [],[]
vmc_hci, dmc_hci = [],[]

tstep = 0.02
determinant_cutoff = 10**(-6)
#mc_tol = [0.1,0.08,0.05,0.02] 
orbitals = ['large'] #'orbitals','fixed',
nconfig = [3200,400,800,1600,6400,12800] #,25600
dmc_nconfig =nconfig[-1]
statenumber = [0]# ,1

for molecule,basis,length in prod:

    dir_hf = f"{molecule}_{length}/hf/{basis}"
    hf_target.append(f"{dir_hf}/mf.chk")
    cc_target.append(f"{dir_hf}/cc.chk")
    fci_target.append(f"{dir_hf}/fci.chk")

    for t in tol:
        hci_target.append(f"{molecule}_{length}/hf/{basis}/hci{t}.chk")

    
    vmc_target += generate_vmc_targets(dir_hf,nconfig,orbitals)
    dmc_target += generate_dmc_targets(dir_hf,dmc_nconfig,orbitals,tstep)
    hci = generate_hci_targets(dir=dir_hf,nconfig=[nconfig,dmc_nconfig],tol=mc_tol,orbitals=orbitals,determinant_cutoff=determinant_cutoff,tstep=tstep,statenumber=0)
    vmc_hci += hci[1]
    dmc_hci += hci[2]

# targets = vmc_target + vmc_hci + dmc_target + dmc_hci
"""