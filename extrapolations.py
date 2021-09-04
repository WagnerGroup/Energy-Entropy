import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pyqmc.obdm


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


def make_eigenvalue_plot(rdm, method='cc', sigma=0.01, epsilon=0.0):
    fig, axes = plt.subplots(1,3,figsize=(9,3), sharey=True)
    for ax,noise in zip(axes, [0.0, sigma/2.0, sigma]):
        noise = np.sqrt(epsilon**2 + noise**2)
        w, s = compute_entropy(rdm, noise)
        # if len(rdm.shape)==2: 
        circle1 = plt.Circle((0, 0), noise*np.sqrt(rdm.shape[1]), color='r', alpha=0.2)
        ax.add_patch(circle1)
        ax.scatter(w.real, w.imag, label=f"{noise}",s=2)

        ax.set_title(f"{noise}")
        ax.set_xlim(-0.1,0.3)
        ax.set_xlabel("Re $\lambda_i$")
    axes[0].set_ylabel("Im $\lambda_i$")
    plt.savefig("("+method+")entropy_eigenvalue_plot.pdf", bbox_inches='tight')


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


def plot_entropy_noise(df,method='cc'):
    plt.figure(figsize=(3,3))
    sns.regplot(x='x',y='symmetrize', label='Symmetrize', data=df)
    sns.regplot(x='x',y='filter',label='Enforce positivity', data=df)
    sns.regplot(x='x',y='aggressive', label='Circle reject', data=df)

    plt.legend()
    plt.xlabel('Noise x')
    plt.ylabel("Estimated entropy per atom")
    plt.savefig("("+method+")entropy_noise.pdf",bbox_inches='tight')

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
    vmc_dm, vmc_dm_err = read_rdm(vmc_fname)
    extrapolated_dm = 2 * mixed_dm - vmc_dm
    extrapolated_dm_err = np.sqrt( 4 * mixed_dm_err**2 + vmc_dm_err**2)
    return extrapolated_dm, extrapolated_dm_err

