import h5py
import numpy as np
from scipy import stats
import pyqmc.obdm
import pandas as pd
import os.path
from os import path
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

def avg(data):
    mean = np.mean(data,axis=0)
    error = np.std(data,axis=0)/np.sqrt(len(data)-1) # compute standard deviation of the mean, not just of samples
    return mean,error

def normalize_rdm(rdm1_value,rdm1_norm,warmup):
    rdm1, rdm1_err = avg(rdm1_value[warmup:,...])
    rdm1_norm, rdm1_norm_err = avg(rdm1_norm[warmup:,...])
    rdm1 = pyqmc.obdm.normalize_obdm(rdm1,rdm1_norm)
    rdm1_err = pyqmc.obdm.normalize_obdm(rdm1_err,rdm1_norm) 
    return rdm1, rdm1_err

def calculate_trace(dm):
    if len(dm.shape) == 2:
        dm = np.asarray([dm/2.0,dm/2.0])
    t = dm - np.matmul(dm,dm)
    u,v = np.linalg.eig(t)
    # u = u[u>0]
    return np.sum(u).real

def calculate_entropy(dm):
    if len(dm.shape) == 2:
        dm = np.asarray([dm/2.0,dm/2.0])
    u,v = np.linalg.eig(dm)
    # print(u)
    u = u[u>0]
    return -np.sum(np.log(u)*u).real


def old_one(dm, dm_err, pts = 100):
    epsilon = np.mean(dm_err)
    sigma = 5 * epsilon
    noises = np.linspace(0, sigma, pts)
    entropies = []
    for noise in noises:
        our_dm = dm + np.random.normal(scale=noise, size=dm.shape)
        u,v = np.linalg.eig(our_dm)
        u = u[u>0]
        entropies.append(-np.sum(np.log(u)*u).real)
    return np.sqrt(epsilon**2 + noises**2), np.asarray(entropies)


def calculate_entropy_with_added_noise(dm, dm_err, pts=10):
    epsilon = np.mean(dm_err)
    # dm = dm + np.random.normal(scale=epsilon, size=dm.shape)
    sigma = 5 * epsilon
    noises = np.linspace(0, sigma, 10)
    e, entropies = [], []
    for noise in noises:
        for i in range(10):
            our_dm = dm + np.random.normal(scale=noise, size=dm.shape)
            u,v = np.linalg.eig(our_dm)
            u = u[u>0]
            e.append(-np.sum(np.log(u)*u).real)
        entropies.append(np.mean(e))
    # print(len(entropies))
    return np.sqrt(epsilon**2 + noises**2), np.asarray(entropies)


def fit_noises(fname, dm, dm_err, draw, N):
    # noises, computed_entropy = calculate_entropy_with_added_noise(dm, dm_err)  # 
    noises, computed_entropy = old_one(dm, dm_err)
    data = {}
    data['x'] = noises
    data['y'] = computed_entropy
    fit = smf.ols("y ~ x", data = data).fit()
    entropy_without_noise = fit.params['Intercept']
    entropy_err = fit.bse[0]
    # print("Results:",fit.params['Intercept']/2, entropy_err/2)

    data['x'] = np.append([0], data['x'])
    data['y'] = np.append([entropy_without_noise], data['y'])
    # print(data['x'][-1])

    if draw:
        c = {'vmc': '#1f77b4', 'dmc': '#ff7f0e'}
        method = fname.split('/')[-1][:3]
        sns.regplot(data['x'], data['y']/N, color=c[method], label=fname, scatter_kws={'s':20})
        # plt.plot(data['x'], fit.predict(data)/2, label='fit')
        plt.errorbar(0, entropy_without_noise/N, yerr= entropy_err/N, color=c[method], marker='x', markersize=8, label=str(entropy_without_noise/N))
    return entropy_without_noise, entropy_err


def read_vmc(fname, flag = True, draw=False, N=2): # False if called by read_dmc
    with h5py.File(fname,'r') as f:
        # print(list(f.keys()))
        warmup = 2
        energy = f['energytotal'][warmup:,...]
        e_tot,error = avg(energy)
    
        rdm1_up,rdm1_up_err = normalize_rdm(f['rdm1_upvalue'],f['rdm1_upnorm'],warmup)
        rdm1_down,rdm1_down_err = normalize_rdm(f['rdm1_downvalue'],f['rdm1_downnorm'],warmup)
        rdm1 = np.array([rdm1_up,rdm1_down])
        rdm1_err = np.array([rdm1_up_err, rdm1_down_err])
        entropy = calculate_entropy(rdm1)
        if not flag: 
            return e_tot, error, entropy, rdm1, rdm1_err
        else: 
            entropy_without_noise, entropy_err = fit_noises(fname, rdm1, rdm1_err, draw, N) 
            return e_tot, error, entropy, entropy_without_noise, entropy_err, calculate_trace(rdm1)     
    

def read_dmc(fname, draw=False, N=2):
    e_tot, error, mixed_entropy, rdm1, rdm1_err = read_vmc(fname, False)
    vmc_fname = fname.replace("dmc","vmc")
    vmc_fname = vmc_fname.replace("_0.02.chk",".chk")
        
    if path.exists(vmc_fname):
        # print(vmc_fname)
        vmc_dm, vmc_dm_err = read_vmc(vmc_fname, False)[-2:]
        extrapolated_dm = 2 * rdm1 - vmc_dm
        extrapolated_dm_err = np.sqrt( 4 * rdm1_err**2 + vmc_dm_err**2)
        extrapolated_entropy = calculate_entropy(extrapolated_dm)
        tr = calculate_trace(extrapolated_dm)
        # print(mixed_entropy, extrapolated_entropy)
        assert extrapolated_entropy > 0
        entropy_without_noise, entropy_err = fit_noises(fname, extrapolated_dm,
                                                            extrapolated_dm_err, draw, N) 
    else: # means vmc result is missing 
        extrapolated_entropy, entropy_without_noise = -1, -1 
        baseline_noise, entropy_err = 0, 0
        tr = calculate_trace(rdm1)
    return e_tot, error, mixed_entropy, extrapolated_entropy, entropy_without_noise, entropy_err, tr

def plot(N, vmc_files, dmc_files):
    fig,ax = plt.subplots(1,1) # ,figsize=(3,3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    _ = read_vmc(vmc_files, True, True, N)
    _ = read_dmc(dmc_files, True, N)

    plt.legend(loc=2, fontsize=10)
    plt.rcParams.update({'font.size':10})
    # plt.xlim(-0.00005,0.005)
    plt.xlabel("x", fontsize=10)
    plt.ylabel("s (x)", fontsize=10)
    plt.tight_layout()
    # plt.title("System:"+vmc_files[:6])
    plt.savefig("Plots/Noise("+vmc_files[:6]+")_old.pdf")
    return


if __name__=="__main__":

    vmc_files = "h6_2.0/vtz/vmc_hci0.1_0_large_0_12800.chk"
    dmc_files = "h6_2.0/vtz/dmc_hci0.1_0_large_0_12800_0.02.chk"

    plot(6, vmc_files, dmc_files)

    vmc_files = "h2_2.0/vtz/vmc_hci0.1_0_large_0_3200.chk"
    dmc_files = "h2_2.0/vtz/dmc_hci0.1_0_large_0_3200_0.02.chk"

    # plot(2, vmc_files, dmc_files)

    