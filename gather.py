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
    if "dmc" in method:
        method += spl[-1]
    if "hci" in startingwf:
        tol = startingwf[3:]
        startingwf = "hci"
    else:
        print("exception")

    return method, startingwf, determinant_cutoff, tol

def extract_from_fname(fname):
    fname = fname.replace('.chk','')
    spl = fname.split('/')
    spl_1 = spl[0].split('_')

    if '_' in spl[3]:
        spl_2 = spl[3].split('_')
        method, startingwf, determinant_cutoff, tol = separate_variables_in_mcfname(spl_2)
        if (startingwf == "mf"):
            startingwf = "sj"
    else: 
        startingwf = spl[1]
        tol, determinant_cutoff = "/", "/"
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
            "det_cutoff": determinant_cutoff,
            }


def track_opt_determinants(fname):
    """ 
    """
    if 'vmc' in fname:
        opt_fname = fname.replace("vmc","opt")
    elif 'dmc' in fname:
        opt_fname = fname.replace("dmc","opt")
        variables = fname.split('/')[-1].split('_')[1:]
        opt_fname = opt_fname.replace("_"+variables[-1],".chk")
    else:
        opt_fname = fname
    opt_nblock = opt_fname.split('/')[-1].split('.chk')[0].split('_')[-1]
    with h5py.File(opt_fname,'r') as f:
        # print(list(f.keys()))
        if 'wf1det_coeff' in list(f['wf'].keys()):
            determinants = np.array(f['wf']['wf1det_coeff']).shape[0]
        else:
            determinants = 1
        df = pyq.read_opt(opt_fname)
        opt_energy = np.array(df['energy'])[-1]
        opt_err  = np.array(df['error'])[-1]
    return determinants, opt_nblock, opt_energy, opt_err


################################### compute related-properties ########################

def store_hf_energy(fname):
    spl = fname.split('/')
    folder = fname.replace(spl[-1],"mf.chk")
    with h5py.File(folder,'r') as f:
        e_hf = f['scf']['e_tot'][()]
    return e_hf


def compute_entropy_aggressive(rdm, noise=0.0, epsilon=0.0):
    """
    Args:
        rdm: 1-RDM matrix elements
        epsilon: Average statistical uncertainty of the 1-RDM matrix elements
        noise: Standard deviation of random variable added to 1-RDM matrix elements
    Returns:
        entropy_min: entropy
        entropy_max: an upper bound for entropy
    """
    if len(rdm.shape) == 2:
        rdm = np.asarray([rdm/2.0,rdm/2.0])

    dm = rdm + np.random.randn(*rdm.shape)*noise
    w = np.linalg.eigvals(dm)
    radius = np.sqrt(epsilon**2+noise**2)*np.sqrt(rdm.shape[1]) ###
    wr = w[ np.abs(w)>radius ].real
    wr = wr[wr>0.0]
    num_exclude = np.sum(wr>0.0)
    trace = np.sum(wr)
    trace_missing = np.ceil(trace)-trace
    w_worst = np.ones(num_exclude)*trace_missing/num_exclude
    missing_entropy = -np.sum(w_worst*np.log(w_worst))
    print("excluded ", num_exclude, 'trace missing', trace_missing, 'maximum missing entropy',missing_entropy )
    entropy_min = -np.sum(wr*np.log(wr))
    entropy_max = entropy_min + missing_entropy
    return np.asarray((entropy_min, entropy_max))

################################### read rdm and entropy from QMC ########################

def read_rdm(fname, warmup = 50, reblock = None):
    """ read rdm from vmc chkfiles, no need for warmup and reblock
    """
    dat = pyq.read_mc_output(fname, warmup, reblock)
    print(list(dat.keys()))
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

def extrapolate_rdm(fname, warmup = 50, reblock = 20):
    """ read rdm from dmc chkfiles
    """
    mixed_dm, mixed_dm_err = read_rdm(fname, warmup, reblock)
    vmc_fname = change_to_vmc_fname(fname)
    if path.exists(vmc_fname):
        vmc_dm, vmc_dm_err = read_rdm(vmc_fname)
        extrapolated_dm = 2 * mixed_dm - vmc_dm
        extrapolated_dm_err = np.sqrt( 4 * mixed_dm_err**2 + vmc_dm_err**2)
        return extrapolated_dm, extrapolated_dm_err
    else: 
        print("Missing VMC")
        return mixed_dm, mixed_dm_err

def read_mc(fname):
    """
    """
    nblocks, nsteps = "/", "/"
    warmup = 50
    if 'vmc' in fname:
        dat = pyq.read_mc_output(fname, warmup)
        e_tot, error = dat['energytotal'], dat['energytotal_err']
        rdm1, rdm1_err = read_rdm(fname)
        with h5py.File(fname,'r') as f:
            nblocks = len(f['block'])
    elif 'dmc' in fname:
        reblock = 20   
        dat = pyq.read_mc_output(fname, warmup, reblock)
        e_tot, error = dat['energytotal'], dat['energytotal_err']
        rdm1, rdm1_err = extrapolate_rdm(fname, warmup, reblock)
        with h5py.File(fname,'r') as f:
            tstep = np.array(f['tstep'])[0]
            branchtime = np.array(f['nsteps'])[0]
            # print(tstep, branchtime)
            nsteps = int(len(f['step'])*tstep*branchtime)
    else: 
        print("exception")
    epsilon = np.mean(rdm1_err)
    entropy_min, entropy_max = compute_entropy_aggressive(rdm1, epsilon=epsilon)
    return e_tot, error, epsilon, entropy_min, entropy_max, nblocks, nsteps
    


################################### gather ########################################

def read(fname, method):
    e_tot, error = 0.0, 0.0
    entropy_min, entropy_max = 0.0, 0.0
    determinants = "/"
    # e_corr, trace = 0.0, 0.0
    opt_nblocks, vmc_nblocks, dmc_nsteps = "/", "/", "/"
    rdm_err = 0.0
    if 'opt' in method:
        determinants, opt_nblocks, opt_e, opt_err = track_opt_determinants(fname)
        e_tot, error = opt_e, opt_err 
    elif 'mc' in method:
        determinants, opt_nblocks, opt_e, opt_err = track_opt_determinants(fname)
        e_tot, error, rdm_err, entropy_min, entropy_max, vmc_nblocks, dmc_nsteps = read_mc(fname)
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
                entropy_min, _ = compute_entropy_aggressive(rdm1)
                entropy_max = entropy_min
                # trace = compute_trace(rdm1)
            elif 'hci' in method:
                e_tot = np.array(f['ci']['energy'])[0] 
                determinants = f['ci']['_strs'][()].shape[0]
                rdm1 = np.array(f['ci']['rdm'])
                entropy_min, _ = compute_entropy_aggressive(rdm1)
                entropy_max = entropy_min
                # trace = compute_trace(rdm1)

    # e_hf = store_hf_energy(fname)
    # e_corr = e_hf - e_tot
    return determinants, opt_nblocks, e_tot, error, rdm_err, entropy_min, entropy_max, vmc_nblocks, dmc_nsteps #, e_corr, trace

def create(fname):
    print(fname)
    record = extract_from_fname(fname)
    N = 1
    if record["molecule"][0] == 'h':
        N = int(record["molecule"][1:])
    method = record["method"]
    determinants, opt_nblocks, e_tot, error, rdm_err, entropy_min, entropy_max, vmc_nblocks, dmc_nsteps = read(fname, method)
    record.update({
           "ndet": determinants,
           "opt_nblocks": opt_nblocks, 
           "vmc_nblocks": vmc_nblocks,
           "dmc_nsteps": dmc_nsteps, 
           "energy/N": e_tot/N,
           "error/N": error/N,
           "rdm_err/N": rdm_err/N, 
           "entropy_min/N": entropy_min/N,
           "entropy_max/N": entropy_max/N,
    })
    return record


if __name__=="__main__":

    fname = []
    for name in glob.glob('**/*.chk',recursive=True):
        if "archive" in name:
            continue
        if name[0] != 'h':
            continue
        fname.append(name)

    df = pd.DataFrame([create(name) for name in fname])
    df = df.sort_values(by=['molecule','bond_length','hci_tol','det_cutoff','method','opt_nblocks'])
    df.to_csv("data.csv", index=False)