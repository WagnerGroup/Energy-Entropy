import h5py
import numpy as np
from scipy import stats
import pyqmc.api as pyq
import pyqmc.obdm
import itertools
import pandas as pd
import glob 
import os.path
from os import path
# import gather_entropy
# import extrapolations
# from extrapolations import avg, read_rdm, extrapolate_rdm, compute_entropy_aggressive, compute_trace

################################### read general variables from filenames ########################

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


################################### compute related-properties ########################

def store_hf_energy(fname):
    spl = fname.split('/')
    folder = fname.replace(spl[-1],"mf.chk")
    with h5py.File(folder,'r') as f:
        e_hf = f['scf']['e_tot'][()]
    return e_hf

def compute_entropy_aggressive(rdm, epsilon=0.0, noise=0.0):
    if len(rdm.shape) == 2:
        rdm = np.asarray([rdm/2.0,rdm/2.0])

    dm = rdm + np.random.randn(*rdm.shape)*noise
    w = np.linalg.eigvals(dm)
    radius = np.sqrt(epsilon**2+noise**2)*np.sqrt(rdm.shape[1]) ###
    wr = w[ np.abs(w)>radius ].real 
    wr = wr[wr>0.0]
    return w, -np.sum(wr*np.log(wr))

################################### read rdm and entropy from QMC ########################

def read_rdm(fname, warmup=2, reblock=20):
    dat = pyq.read_mc_output(fname, warmup, reblock)
    rdm1_up = pyqmc.obdm.normalize_obdm(dat['rdm1_upvalue'], dat['rdm1_upnorm'])
    rdm1_up_err = pyqmc.obdm.normalize_obdm(dat['rdm1_upvalue_err'], dat['rdm1_upnorm'])
    rdm1_down = pyqmc.obdm.normalize_obdm(dat['rdm1_downvalue'], dat['rdm1_downnorm'])
    rdm1_down_err = pyqmc.obdm.normalize_obdm(dat['rdm1_downvalue_err'], dat['rdm1_downnorm'])
    rdm1 = np.array([rdm1_up,rdm1_down])
    rdm1_err = np.array([rdm1_up_err, rdm1_down_err])
    return rdm1, rdm1_err

def change_to_vmc_fname(dmc_fname):
    vmc_fname = dmc_fname.replace("dmc","vmc")
    variables = dmc_fname.split('/')[-1].split('_')[1:]
    if "0.02" in variables[-1]: 
        vmc_fname = vmc_fname.replace("_"+variables[-1],".chk")
    else: 
        vmc_fname = vmc_fname.replace("_0.02","")
        vmc_fname = vmc_fname.replace("_"+variables[-1],".chk")
    return vmc_fname

def extrapolate_rdm(fname,warmup = 2, reblock = 20):
    mixed_dm, mixed_dm_err = read_rdm(fname,warmup)
    vmc_fname = change_to_vmc_fname(fname)
    if path.exists(vmc_fname):
        vmc_dm, vmc_dm_err = read_rdm(vmc_fname, warmup, reblock)
        extrapolated_dm = 2 * mixed_dm - vmc_dm
        extrapolated_dm_err = np.sqrt( 4 * mixed_dm_err**2 + vmc_dm_err**2)
        return extrapolated_dm, extrapolated_dm_err
    else: 
        print("Missing VMC")
        return mixed_dm, mixed_dm_err

def read_vmc(fname, warmup=2, reblock=20):
    with h5py.File(fname,'r') as f:
        blocks = len(f['block'])
    
    dat = pyq.read_mc_output(fname, warmup, reblock)
    e_tot, error = dat['energytotal'], dat['energytotal_err']
    rdm1, rdm1_err = read_rdm(fname, warmup, reblock)
    _, entropy = compute_entropy_aggressive(rdm1, epsilon=np.mean(rdm1_err))
    return e_tot, error, entropy, blocks #, trace_object

def read_dmc(fname, warmup=2, reblock=20):
    with h5py.File(fname,'r') as f:
        tstep = np.array(f['tstep'])[0]
        branchtime = np.array(f['nsteps'])[0]
        # print(tstep, branchtime)
        nsteps = int(len(f['step'])*tstep*branchtime)
    dat = pyq.read_mc_output(fname, warmup, reblock)
    e_tot, error = dat['energytotal'], dat['energytotal_err']
    rdm1, rdm1_err = extrapolate_rdm(fname, warmup, reblock)
    _, entropy = compute_entropy_aggressive(rdm1, epsilon=np.mean(rdm1_err))
    # trace_object = compute_trace(rdm1)
    return e_tot, error, entropy, nsteps #, trace_object

################################### gather ########################################

def read(fname, method):
    e_tot, error, entropy = 0.0, 0.0, 0.0
    determinants = "/"
    # e_corr, trace = 0.0, 0.0
    blocks, nsteps = "/", "/"

    if 'vmc' in method:
        e_tot, error, entropy, blocks = read_vmc(fname)
        determinants = track_opt_determinants(fname)
    elif 'dmc' in method:
        e_tot, error, entropy, nsteps = read_dmc(fname)
        determinants = track_opt_determinants(change_to_vmc_fname(fname))
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
                _, entropy = compute_entropy_aggressive(rdm1)
                # trace = compute_trace(rdm1)
            elif 'hci' in method:
                e_tot = np.array(f['ci']['energy'])[0] 
                determinants = f['ci']['_strs'][()].shape[0]
                rdm1 = np.array(f['ci']['rdm'])
                _, entropy = compute_entropy_aggressive(rdm1)
                # trace = compute_trace(rdm1)

    # e_hf = store_hf_energy(fname)
    # e_corr = e_hf - e_tot
    return determinants, e_tot, error, entropy, blocks, nsteps #, e_corr, trace

def create(fname):
    record = extract_from_fname(fname)
    N = 1
    if record["molecule"][0] == 'h':
        N = int(record["molecule"][1:])
    method = record["method"]
    determinants, e_tot, error, entropy, blocks, nsteps = read(fname, method)
    record.update({
           "blocks": blocks,
           "nsteps": nsteps, 
           "ndet": determinants,
           "natom": N,
           "energy/N": e_tot/N,
           "error/N": error/N,
           "entropy/N": entropy/N,
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