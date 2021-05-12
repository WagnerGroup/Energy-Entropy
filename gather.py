import h5py
import numpy as np
from scipy import stats
import pyqmc.obdm
import itertools
import pandas as pd
import glob 

def separate_variables_in_fname(spl):
    method=spl[0]
    startingwf=spl[1]
    i=1
    if "hci" in startingwf:
        i+=1
    orbitals=spl[i+1]
    statenumber=spl[i+2]
    nconfig=spl[i+3]
    if "dmc" in method:
        method+=spl[i+4]
    return method,startingwf,orbitals,statenumber,nconfig

def extract_from_fname(fname):
    fname=fname.replace('.chk','')
    spl=fname.split('/')
    if '_' in spl[3]:
        spl_2=spl[3].split('_')
        method,startingwf,orbitals,statenumber,nconfig=separate_variables_in_fname(spl_2)
        if (startingwf=="mf"):
            startingwf=spl[1]
    else: 
        startingwf=spl[1]
        orbitals,nconfig="/","/"
        statenumber=0
        method=spl[3]
        if (method=="mf"):
            method=spl[1]

    return {"bond_length": spl[0][3:],
            "basis":spl[2],
            "startingwf":startingwf,
            "method":method,
            "orbitals":orbitals,
            "statenumber":statenumber,
            "determinant_cutoff":0,
            "nconfig":nconfig
            }


def avg(data, reblock=16, correlated=True):
    if (correlated == False):
        mean=np.mean(data,axis=0)
        error=np.std(data,axis=0)/np.sqrt(len(data)-1)
    else:
        vals = pyqmc.reblock.reblock(data, reblock)
        mean=np.mean(data,axis=0)
        error=scipy.stats.sem(vals,axis=0)
    return mean,error

def calculate_entropy(dm):
    if len(dm.shape) == 2:
        dm = np.asarray([dm/2.0,dm/2.0])
    u,v = np.linalg.eig(dm)
    #print(u)
    u = u[u>0]
    return -np.sum(np.log(u)*u).real

def normalize_rdm(rdm1_value,rdm1_norm,warmup):
    rdm1, rdm1_err=avg(rdm1_value[warmup:,...])
    rdm1_norm, rdm1_norm_err = avg(rdm1_norm[warmup:,...])
    rdm1=pyqmc.obdm.normalize_obdm(rdm1,rdm1_norm)
    rdm1_err=pyqmc.obdm.normalize_obdm(rdm1_err,rdm1_norm) 
    return rdm1,rdm1_err
    
def read_vmc(fname):
    with h5py.File(fname,'r') as f:
        #print(list(f.keys()))
        warmup=2
        energy=f['energytotal'][warmup:,...]
        e_tot,error=avg(energy, correlated=True)
    
        rdm1_up,rdm1_up_err=normalize_rdm(f['rdm1_upvalue'],f['rdm1_upnorm'],warmup)
        rdm1_down,rdm1_down_err=normalize_rdm(f['rdm1_downvalue'],f['rdm1_downnorm'],warmup)
        rdm1=np.array([rdm1_up,rdm1_down])
        entropy=calculate_entropy(rdm1)
        return e_tot,error,entropy

def read_dmc(fname):
    return read_vmc(fname)

def read(fname):
    with h5py.File(fname,'r') as f:
        e_tot,error,entropy=0.0,0.0,0.0
        method=extract_from_fname(fname)["method"]
        if 'cc' in method:
            e_tot=f['ccsd']['energy'][()]
        elif 'vmc' in method:
            e_tot,error,entropy=read_vmc(fname)
        elif 'dmc' in method:
            e_tot,error,entropy=read_dmc(fname)
        elif 'hf' in method: 
            e_tot = f['scf']['e_tot'][()]
        elif 'fci' in method:
            e_tot=np.array(f['e_tot'][()])[0] #state0,1,2,3,
        elif 'hci' in method:
            e_tot=np.array(f['ci']['energy'])[0] #state0,1,2,3
        return e_tot,error,entropy

def create(fname):
    e_tot,error,entropy = read(fname)
    record = extract_from_fname(fname)
    record.update({
           "energy":e_tot,
           "error":error,
           "entropy":entropy
    })
    return record

if __name__=="__main__":
    fname=[]
    fqmc=[]
    fhci=[]
    for name in glob.glob('**/*.chk',recursive=True):
        method=extract_from_fname(name)["method"]
        startingwf=extract_from_fname(name)["startingwf"]
        if 'opt' in method:
            continue
        if ('hci' in method or 'hci' in startingwf):
            fhci.append(name)
        if ('vmc' in method or 'dmc' in method):
            fqmc.append(name)
        fname.append(name)

    df=pd.DataFrame([create(name) for name in fname])
    df.to_csv("h2_data.csv", index=False)

    df1=pd.DataFrame([create(name) for name in fqmc])
    df1 = df1[(df1.basis=='vtz')]
    df1.to_csv("h2_qmc.csv", index=False)

    #df2=pd.DataFrame([create(name) for name in fhci])
    #df2.to_csv("hci.csv", index=False)
    
