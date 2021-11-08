import h5py
import numpy as np
from scipy import stats
import pyqmc.obdm
import itertools
import pandas as pd
import glob 
# import gather_entropy
import extrapolations

def separate_variables_in_mcfname(spl):
    method = spl[0]
    startingwf = spl[1]
    determinant_cutoff, tol = "/", "/"

    i = 1
    if "hci" in startingwf:
        i += 1
        determinant_cutoff = spl[i]

    orbitals = spl[i+1]
    statenumber = spl[i+2]
    nconfig = int(spl[i+3])

    if "dmc" in method:
        method += spl[i+4]

    if "hci" in startingwf:
        method += "_hci"
        tol = startingwf[3:]
    elif "mf" in startingwf:
        method += "_sj"
    else:
        print("check1")

    return method, startingwf, orbitals, statenumber, nconfig, determinant_cutoff, tol

def extract_from_fname(fname):
    fname = fname.replace('.chk','')
    spl = fname.split('/')
    spl_1 = spl[0].split('_')

    if '_' in spl[3]:
        spl_2 = spl[3].split('_')
        method,startingwf,orbitals,statenumber,nconfig, determinant_cutoff, tol = separate_variables_in_mcfname(spl_2)
        if (startingwf == "mf"):
            startingwf = spl[1]
    else: 
        startingwf = spl[1]
        orbitals, nconfig, tol, determinant_cutoff = "/", "/", "/", "/"
        statenumber = 0
        method = spl[3]
        if "hci" in method:
            tol = method[3:]
            method = "hci"
        if method == "mf":
            method = spl[1]

    return {"molecule": spl_1[0],
            "bond_length": spl_1[1],
            "basis": spl[2],
            "startingwf": startingwf,
            "method": method,
            "hci_tol": tol,
            "orbitals": orbitals,
            "statenumber": statenumber,
            "determinant_cutoff": determinant_cutoff,
            "nconfig": nconfig
            }


def track_opt_determinants(fname):
    fname = fname.replace("vmc","opt")
    # print(fname)
    with h5py.File(fname,'r') as f:
        # print(list(f.keys()))
        if 'wf1det_coeff' in list(f['wf'].keys()):
            determinants = np.array(f['wf']['wf1det_coeff']).shape[0]
        else:
            print("Single determinants:", fname)
            determinants = 1
    return determinants #, it


def read_vmc(fname, warmup=2):
    with h5py.File(fname,'r') as f:
        energy = f['energytotal'][warmup:,...]
        e_tot,error = extrapolations.avg(energy)
    
    rdm1, rdm1_err = extrapolations.read_rdm(fname,warmup)
    _, entropy = extrapolations.compute_entropy_aggressive(rdm1, epsilon=np.mean(rdm1_err))
    trace_object = extrapolations.compute_trace(rdm1)
    return e_tot, error, entropy, trace_object #errorbar?

def read_dmc(fname, warmup=2):
    with h5py.File(fname,'r') as f:
        energy = f['energytotal'][warmup:,...]
        e_tot,error = extrapolations.avg(energy)
    
    rdm1, rdm1_err = extrapolations.extrapolate_rdm(fname,warmup)
    _, entropy = extrapolations.compute_entropy_aggressive(rdm1, epsilon=np.mean(rdm1_err))
    trace_object = extrapolations.compute_trace(rdm1)
    return e_tot, error, entropy, trace_object

def store_hf_energy(fname):
    spl = fname.split('/')
    folder = fname.replace(spl[-1],"mf.chk")
    with h5py.File(folder,'r') as f:
        e_hf = f['scf']['e_tot'][()]
    return e_hf

def read(fname, method):
    e_tot, error, entropy = 0.0, 0.0, 0.0
    determinants = "/"
    e_corr, trace = 0.0, 0.0
    if 'vmc' in method:
        e_tot, error, entropy, trace = read_vmc(fname)
        determinants = track_opt_determinants(fname)
    elif 'dmc' in method:
        e_tot, error, entropy, trace = read_dmc(fname)
    else: 
        with h5py.File(fname,'r') as f:
            if 'hf' in method: 
                e_tot = f['scf']['e_tot'][()]
                determinants = 1
            elif 'fci' in method:
                e_tot = np.array(f['e_tot'][()])[0] #state_0,1,2,3, 
            
            elif 'cc' in method:
                e_tot = f['ccsd']['energy'][()]
                rdm1 = np.array(f['ccsd']['rdm'])
                _, entropy = extrapolations.compute_entropy_aggressive(rdm1)
                trace = extrapolations.compute_trace(rdm1)
            elif 'hci' in method:
                e_tot = np.array(f['ci']['energy'])[0] 
                determinants = f['ci']['_strs'][()].shape[0]
                rdm1 = np.array(f['ci']['rdm'])
                _, entropy = extrapolations.compute_entropy_aggressive(rdm1)
                trace = extrapolations.compute_trace(rdm1)

    e_hf = store_hf_energy(fname)
    e_corr = e_hf - e_tot
    return determinants, e_tot, error, entropy, e_corr, trace

def create(fname):
    record = extract_from_fname(fname)
    N = 1
    if record["molecule"][0] == 'h':
        N = int(record["molecule"][1:])
    method = record["method"]
    determinants, e_tot, error, entropy, e_corr, trace = read(fname, method)
    record.update({
           "ndet": determinants,
           "natom": N,
           "energy/N": e_tot/N,
           "error/N": error/N,
           "entropy/N": entropy/N,
           "corrE/N": e_corr/N, 
           "rdm1_trace/N": trace/N
    })
    return record


if __name__=="__main__":

    fname = []
    for name in glob.glob('**/*.chk',recursive=True):
        if "opt" in name:
            continue
        if name[0] != 'h':
            continue
        fname.append(name)

    df = pd.DataFrame([create(name) for name in fname])
    df = df.sort_values(by=['natom','bond_length','hci_tol','nconfig'])
    df.to_csv("data.csv", index=False)