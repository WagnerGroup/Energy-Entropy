import h5py
import numpy as np
from scipy import stats
import pyqmc.obdm
import itertools
import pandas as pd
import glob 
import gather_entropy

def separate_variables_in_fname(spl):
    method = spl[0]
    startingwf = spl[1]
    determinant_cutoff = 0
    i = 1
    if "hci" in startingwf:
        i += 1
        determinant_cutoff = spl[i]
    orbitals = spl[i+1]
    statenumber = spl[i+2]
    nconfig = spl[i+3]
    if "dmc" in method:
        method += spl[i+4]
    return method,startingwf,orbitals,statenumber,nconfig, determinant_cutoff

def extract_from_fname(fname):
    fname = fname.replace('.chk','')
    spl = fname.split('/')
    spl_1 = spl[0].split('_')
    determinant_cutoff = 0
    if '_' in spl[3]:
        spl_2 = spl[3].split('_')
        method,startingwf,orbitals,statenumber,nconfig, determinant_cutoff = separate_variables_in_fname(spl_2)
        if (startingwf == "mf"):
            startingwf = spl[1]
    else: 
        startingwf = spl[1]
        orbitals,nconfig = "/","/"
        statenumber = 0
        method = spl[3]
        if (method == "mf"):
            method = spl[1]

    return {"molecule": spl_1[0],
            "bond_length": spl_1[1],
            "basis":spl[2],
            "startingwf":startingwf,
            "method":method,
            "orbitals":orbitals,
            "statenumber":statenumber,
            "determinant_cutoff":determinant_cutoff,
            "nconfig":nconfig
            }

def track_opt_determinants(fname):
    # record = extract_from_fname(fname)
    fname = fname.replace("vmc","opt")
    with h5py.File(fname,'r') as f:
        # print(list(f.keys()))
        if 'wf1det_coeff' in list(f['wf'].keys()):
            determinants = np.array(f['wf']['wf1det_coeff']).shape[0]
        else:
            print("opt:", fname)
            determinants = 1
        it = np.array(f['x']).shape[0]
        # x = np.array(f['x']).shape[1]
    return determinants, it

def read(fname, method):
    with h5py.File(fname,'r') as f:
        e_tot, error, entropy = 0.0, 0.0, 0.0
        # rdm1, rdm1_shape = "/", "/"
        determinants, it = "/", "/"
        mixed_entropy = 0.0
        entropy_err, trace = 0.0, 0.0
        if 'hf' in method: 
            e_tot = f['scf']['e_tot'][()]
        elif 'fci' in method:
            e_tot = np.array(f['e_tot'][()])[0] #state_0,1,2,3, 
        elif 'cc' in method:
            e_tot = f['ccsd']['energy'][()]
            rdm1 = np.array(f['ccsd']['rdm'])
            entropy = gather_entropy.calculate_entropy(rdm1)
            trace = gather_entropy.calculate_trace(rdm1)
        elif 'hci' in method:
            e_tot = np.array(f['ci']['energy'])[0] #state_0,1,2 ? 
            rdm1 = np.array(f['ci']['rdm'])
            entropy = gather_entropy.calculate_entropy(rdm1)
            trace = gather_entropy.calculate_trace(rdm1)
        entropy_without_noise = entropy

        if 'vmc' in method:
            e_tot, error, entropy, entropy_without_noise, entropy_err, trace = gather_entropy.read_vmc(fname)
            determinants, it = track_opt_determinants(fname)
        elif 'dmc' in method:
            e_tot, error, mixed_entropy, entropy, entropy_without_noise, entropy_err, trace = gather_entropy.read_dmc(fname)
        
        return determinants, it, e_tot, error, entropy, entropy_without_noise, mixed_entropy, entropy_err, trace

def create(fname):
    record = extract_from_fname(fname)
    N = 1
    if record["molecule"][0] == 'h':
        N = int(record["molecule"][1])
    method = record["method"]
    determinants, it, e_tot, error, entropy, entropy_without_noise, mixed_entropy, entropy_err, trace = read(fname, method)
    record.update({
           "opt_iteration":it,
           "det_coeff": determinants,
           "energy": e_tot,
           "error": error,
           "entropy": entropy,
           "mixed_entropy": mixed_entropy,
           "N": N,
           "energy_per_atom": e_tot/N,
           "error_per_atom": error/N,
           "entropy_per_atom": entropy/N,
           "entropy_per_atom_without_noise": entropy_without_noise/N,
           "entropy_per_atom_errorbar": entropy_err/N,
           "rdm1_trace_per_atom": trace/N
    })
    return record

if __name__=="__main__":
    fname = []
    fname1 = []
    for name in glob.glob('**/*.chk',recursive=True):
        record = extract_from_fname(name)
        # fname.append(name)
        if 'opt' in record["method"]:
            continue
        if record["orbitals"]=='orbitals':
            continue
        if str(record["determinant_cutoff"])=="0.02":
            continue
        fname.append(name)

    # df = pd.DataFrame([create(name) for name in fname])
    # df.to_csv("data.csv", index=False)
    # print(df)
    df = pd.DataFrame([create(name) for name in fname])
    df.to_csv("data.csv", index=False)
