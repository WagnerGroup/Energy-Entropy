import h5py
import numpy as np
import scipy
import pyqmc.api as pyq
import pyqmc.obdm
import itertools
import pandas as pd
import glob 
import os.path
from os import path


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
    """ track the number of determinants coefficients in optimization
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
    """ extract hartree fork energy in order to compute correlation energy
    """
    spl = fname.split('/')
    folder = fname.replace(spl[-1],"mf.chk")
    with h5py.File(folder,'r') as f:
        e_hf = f['scf']['e_tot'][()]
    return e_hf

def compute_trace_object(rdm, rdm_err=None):
    """ compute Lambda = Tr(rdm-rdm**2)
        TODO: rdm_err
    """
    if len(rdm.shape) == 2:
        rdm = np.asarray([rdm/2.0,rdm/2.0])
    t = rdm - np.matmul(rdm,rdm)
    u,v = np.linalg.eig(t)
    # u = u[u>0]
    return np.sum(u).real


def compute_entropy_aggressive(rdm, noise=0.0, epsilon=0.0, imag_percent=0.01, r_inc_percent=0.01):
    """
    Args:
        rdm: 1-RDM matrix
        epsilon: Averaged statistical uncertainty of the 1-RDM matrix
        noise: Standard deviation of a random matrix added to 1-RDM matrix
    Returns:
        entropy_min: a lower bound for entropy
        entropy_max: an upper bound for entropy
    """
    if len(rdm.shape) == 2:
        rdm = np.asarray([rdm/2.0,rdm/2.0])

    dm = rdm + np.random.randn(*rdm.shape)*noise
    w = np.linalg.eigvals(dm)
    radius = np.sqrt(epsilon**2+noise**2)*np.sqrt(rdm.shape[1]) ###
    wr = w[ np.abs(w)>radius ]
    imag_sum = np.sum(np.abs(wr.imag))
    if imag_sum > radius*imag_percent:
        print("OH NO! We have imaginary components. increasing rejection radius")
        wr_imag_large = wr[np.abs(wr.imag) > radius*imag_percent]
        wr_imag_large = wr_imag_large.tolist()
        for wi in wr_imag_large[::-1]:
            if wi.real>radius:
                wr_imag_large.remove(wi)
        if len(wr_imag_large)>0:
            radius = np.max(np.abs(wr_imag_large))*(1+r_inc_percent)
        wr=wr[np.abs(wr)> radius]

    wr = wr.real
    wr = wr[wr>0.0]
    entropy_baseline = -np.sum(wr*np.log(wr))
    if epsilon==0.0: ### For methods other than QMC
        return entropy_baseline, entropy_baseline
    trace = np.sum(wr)
    trace_missing = np.ceil(trace)-trace
    
    fewest_num_cutoff = np.int(np.ceil(trace_missing/radius))
    w_best = np.ones(fewest_num_cutoff)*trace_missing/fewest_num_cutoff
    missing_entropy_min = -np.sum(w_best*np.log(w_best))
    entropy_min = entropy_baseline + missing_entropy_min

    num_exclude = np.sum(np.abs(w)<radius)
    w_worst = np.ones(num_exclude)*trace_missing/num_exclude
    missing_entropy_max = -np.sum(w_worst*np.log(w_worst))
    entropy_max = entropy_baseline + missing_entropy_max
    
    print('fewest_num_cutoff', fewest_num_cutoff, 'excluded', num_exclude, 
            'minimum missing entropy', missing_entropy_min,
            'maximum missing entropy', missing_entropy_max 
        )
    return entropy_min, entropy_max


################################### read rdm and entropy from QMC ########################
def change_to_vmc_fname(dmc_fname):
    vmc_fname = dmc_fname.replace("dmc","vmc")
    variables = dmc_fname.split('/')[-1].split('_')[1:]
    if "0.02" in variables[-1]: 
        vmc_fname = vmc_fname.replace("_"+variables[-1],".chk")
    else: 
        vmc_fname = vmc_fname.replace("_0.02","")
        vmc_fname = vmc_fname.replace("_"+variables[-1],".chk")
    return vmc_fname


def reblock_and_avg(vals, reblock=None):
    if reblock is not None:
            vals = pyqmc.reblock.reblock(vals, reblock)
    mean = np.mean(vals, axis=0)
    err = scipy.stats.sem(vals, axis=0)
    return mean, err


def normalize_obdm(dat):
    rdm1_up = pyqmc.obdm.normalize_obdm(dat['rdm1_upvalue'], dat['rdm1_upnorm'])
    rdm1_up_err = pyqmc.obdm.normalize_obdm(dat['rdm1_upvalue_err'], dat['rdm1_upnorm'])
    rdm1_down = pyqmc.obdm.normalize_obdm(dat['rdm1_downvalue'], dat['rdm1_downnorm'])
    rdm1_down_err = pyqmc.obdm.normalize_obdm(dat['rdm1_downvalue_err'], dat['rdm1_downnorm'])
    rdm1 = np.array([rdm1_up,rdm1_down])
    rdm1_err = np.array([rdm1_up_err, rdm1_down_err])
    return rdm1, rdm1_err


def extrapolate_rdm(dmc_fname, warmup = 50, reblock = 20):
    """ read rdm from dmc chkfiles
    """
    obdm_sample1 = {}
    obdm_sample2 = {}
    with h5py.File(dmc_fname,'r') as f:
        #print(f.keys())
        for k in ['rdm1_upvalue', 'rdm1_downvalue', 'rdm1_upnorm', 'rdm1_downnorm']: #
            vals = f[k][warmup:]
            nblocks = vals.shape[0]
            sample1 = vals[ : nblocks//2]
            sample2 = vals[-nblocks//2: ]
            m1, err1 = reblock_and_avg(sample1, reblock)
            m2, err2 = reblock_and_avg(sample2, reblock)
            obdm_sample1[k] = m1
            obdm_sample1[k + "_err"]  = err1
            obdm_sample2[k] = m2
            obdm_sample2[k + "_err"]  = err2

    dm_s1, dm_err_s1 = normalize_obdm(obdm_sample1)
    dm_s2, dm_err_s2 = normalize_obdm(obdm_sample2)

    vmc_fname = change_to_vmc_fname(dmc_fname)
    if path.exists(vmc_fname):
        vmc_dat = pyq.read_mc_output(vmc_fname, warmup, reblock=None)
        vmc_dm, vmc_dm_err = normalize_obdm(vmc_dat)
        extrapolated_dm_up = dm_s1[0] + dm_s2[0].conj().T - vmc_dm[0]
        extrapolated_dm_down = dm_s1[1] + dm_s2[1].conj().T - vmc_dm[1]
        extrapolated_dm = np.array([extrapolated_dm_up, extrapolated_dm_down])
        extrapolated_dm_err = np.sqrt((dm_err_s1 + dm_err_s2)**2 + vmc_dm_err**2)
        return extrapolated_dm, extrapolated_dm_err
    else: 
        print("Missing VMC")
        return dm_s2, dm_err_s2


def read_mc(fname):
    """
    Args:
        fname: name of chkfile
    Returns:
        e_tot: total ground state energy
        error: error of ground state energy
        epsilon: averaged statistical uncertainty of the 1-RDM matrix
        entropy_min: a lower bound for entropy
        entropy_max: an upper bound for entropy
        nblocks: vmc nblocks
        nsteps: dmc_nsteps
        trace_object: tr(rdm-rdm**2)
    """
    nblocks, nsteps = "/", "/"
    warmup = 50
    if 'vmc' in fname:
        dat = pyq.read_mc_output(fname, warmup) #reblock=None
        e_tot, error = dat['energytotal'], dat['energytotal_err']
        rdm1, rdm1_err = normalize_obdm(dat)
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
        print("Exception")
    epsilon = np.mean(rdm1_err)
    entropy_min, entropy_max = compute_entropy_aggressive(rdm1, epsilon=epsilon)
    trace_object = compute_trace_object(rdm1, rdm1_err)
    return e_tot, error, epsilon, entropy_min, entropy_max, nblocks, nsteps, trace_object
    


################################### gather ########################################

def read(fname, method):
    """
    Args:
        fname: name of chkfile
        method: quantum chemistry methods or QMC
    Returns:
        determinants: 
        opt_nblocks: 'nblocks' set in vmcoptions in optimization
        e_tot: total ground state energy
        error: error of ground state energy
        rdm_err: averaged statistical uncertainty of the 1-RDM matrix
        entropy_min: a lower bound for entropy
        entropy_max: an upper bound for entropy
        vmc_nblocks: number of blocks in VMC
        dmc_nsteps: number of steps in DMC
        e_corr: correlation energy
        trace_object: tr(rdm-rdm**2)
    """
    e_tot, error = 0.0, 0.0
    entropy_min, entropy_max = 0.0, 0.0
    determinants = "/"
    e_corr, trace_object = 0.0, 0.0
    opt_nblocks, vmc_nblocks, dmc_nsteps = "/", "/", "/"
    rdm_err = 0.0

    e_hf = store_hf_energy(fname)
    if 'hf' in method: 
        e_tot = e_hf
        determinants = 1
    elif 'opt' in method:
        determinants, opt_nblocks, opt_e, opt_err = track_opt_determinants(fname)
        e_tot, error = opt_e, opt_err 
    elif 'mc' in method:
        determinants, opt_nblocks, opt_e, opt_err = track_opt_determinants(fname)
        e_tot, error, rdm_err, entropy_min, entropy_max, vmc_nblocks, dmc_nsteps, trace_object = read_mc(fname)
    else: 
        with h5py.File(fname,'r') as f:
            if 'fci' in method:
                e_tot = np.array(f['e_tot'][()])
            elif 'cc' in method:
                e_tot = f['ccsd']['energy'][()]
                rdm1 = np.array(f['ccsd']['rdm'])
                entropy_min, entropy_max = compute_entropy_aggressive(rdm1)
                trace_object = compute_trace_object(rdm1)
            elif 'hci' in method:
                e_tot = np.array(f['ci']['energy'])[0] 
                determinants = f['ci']['_strs'][()].shape[0]
                rdm1 = np.array(f['ci']['rdm'])
                entropy_min, entropy_max = compute_entropy_aggressive(rdm1)
                trace_object = compute_trace_object(rdm1)
    e_corr = e_hf - e_tot
    return determinants, opt_nblocks, e_tot, error, rdm_err, entropy_min, entropy_max, vmc_nblocks, dmc_nsteps, e_corr, trace_object

def create(fname):
    print(fname)
    record = extract_from_fname(fname)
    N = 1
    if record["molecule"][0] == 'h':
        N = int(record["molecule"][1:])
    method = record["method"]
    determinants, opt_nblocks, e_tot, error, rdm_err, entropy_min, entropy_max, vmc_nblocks, dmc_nsteps, e_corr, trace_object = read(fname, method)
    record.update({
           "ndet": determinants,
           "opt_nblocks": opt_nblocks, 
           "vmc_nblocks": vmc_nblocks,
           "dmc_nsteps": dmc_nsteps, 
           "N": N, 
           "energy/N": e_tot/N,
           "error/N": error/N,
           "rdm_err/N": rdm_err/N, 
           "entropy_min/N": entropy_min/N,
           "entropy_max/N": entropy_max/N,
           "trace_object/N": trace_object/N,
           "e_corr/N": e_corr/N,
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
    df = df.sort_values(by=['N','bond_length','hci_tol','det_cutoff','method','opt_nblocks'])
    df.to_csv("data.csv", index=False)