import h5py
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
import pyqmc.obdm
import os.path
from os import path


def compute_entropy(rdm, noise=0.0):
    if len(rdm.shape) == 2:
        rdm = np.asarray([rdm/2.0,rdm/2.0])

    dm = rdm + np.random.randn(*rdm.shape)*noise
    w = np.linalg.eigvals(dm)
    wr = w.real
    wr = wr[wr>0.0]
    return w, -np.sum(wr*np.log(wr))


def compute_entropy_symmetrize(rdm, noise=0.0):
    if len(rdm.shape)==2:
        rdm = np.asarray([rdm/2.0,rdm/2.0])
    
    dm = rdm + np.random.randn(*rdm.shape)*noise
    sym_up = (dm[0] + dm[0].T)/2.0
    sym_down = (dm[1] + dm[1].T)/2.0
    dm = np.array([sym_up,sym_down])

    w = np.linalg.eigvals(dm)
    wr = w[w>0.0]
    return w, -np.sum(wr*np.log(wr))


def compute_entropy_aggressive(rdm, noise=0.0, epsilon=0.0):
    if len(rdm.shape) == 2:
        rdm = np.asarray([rdm/2.0,rdm/2.0])

    dm = rdm + np.random.randn(*rdm.shape)*noise
    w = np.linalg.eigvals(dm)
    radius = np.sqrt(epsilon**2+noise**2)*np.sqrt(rdm.shape[1]) ###
    wr = w[ np.abs(w)>radius ].real 
    wr = wr[wr>0.0]
    return w, -np.sum(wr*np.log(wr))

def generate_entropy_noise(rdm, N=6, epsilon=0.0, sigma=0.01):
    df = []
    for noise in np.linspace(0, sigma, 20):
        w, s = compute_entropy(rdm,noise)
        w, s_symmetrize = compute_entropy_symmetrize(rdm,noise)
        w, s_aggressive = compute_entropy_aggressive(rdm,noise, epsilon)
        df.append({'noise':noise, 'x':np.sqrt(epsilon**2 + noise**2),
                    'filter':s/N, 'symmetrize':s_symmetrize/N, 'aggressive':s_aggressive/N})
    df = pd.DataFrame(df)
    return df


def avg(data):
    mean = np.mean(data,axis=0)
    error = np.std(data,axis=0)/np.sqrt(len(data)-1) 
    return mean,error

def normalize_rdm(rdm1_value,rdm1_norm,warmup):
    rdm1, rdm1_err = avg(rdm1_value[warmup:,...])
    rdm1_norm, rdm1_norm_err = avg(rdm1_norm[warmup:,...])
    rdm1 = pyqmc.obdm.normalize_obdm(rdm1,rdm1_norm)
    rdm1_err = pyqmc.obdm.normalize_obdm(rdm1_err,rdm1_norm) 
    return rdm1, rdm1_err

def read_rdm(fname, warmup=2):
    with h5py.File(fname,'r') as f:
        rdm1_up,rdm1_up_err = normalize_rdm(f['rdm1_upvalue'],f['rdm1_upnorm'],warmup)
        rdm1_down,rdm1_down_err = normalize_rdm(f['rdm1_downvalue'],f['rdm1_downnorm'],warmup)
        rdm1 = np.array([rdm1_up,rdm1_down])
        rdm1_err = np.array([rdm1_up_err, rdm1_down_err])
    return rdm1, rdm1_err

def extrapolate_rdm(fname,warmup=2):
    mixed_dm, mixed_dm_err = read_rdm(fname,warmup)
    vmc_fname = fname.replace("dmc","vmc")
    vmc_fname = vmc_fname.replace("_0.02.chk",".chk")

    if path.exists(vmc_fname):
        vmc_dm, vmc_dm_err = read_rdm(vmc_fname)
        extrapolated_dm = 2 * mixed_dm - vmc_dm
        extrapolated_dm_err = np.sqrt( 4 * mixed_dm_err**2 + vmc_dm_err**2)
        return extrapolated_dm, extrapolated_dm_err
    else: 
        print("Missing VMC")
        return mixed_dm, mixed_dm_err
### also compute other properties

def compute_trace(dm):
    if len(dm.shape) == 2:
        dm = np.asarray([dm/2.0,dm/2.0])
    t = dm - np.matmul(dm,dm)
    u,v = np.linalg.eig(t)
    # u = u[u>0]
    return np.sum(u).real

