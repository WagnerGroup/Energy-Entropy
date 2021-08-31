import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pyscf
import pyqmc.recipes
import h5py
import pyqmc.api as pyq
import pyqmc.tbdm
import pyscf.hci
import pyscf.cc
from functools import partial

def save_scf_iteration(chkfile, envs):
    cycle = envs['cycle']
    info = {'mo_energy':envs['mo_energy'],
            'e_tot'   : envs['e_tot']}
    pyscf.scf.chkfile.save(chkfile, 'iteration/%d' % cycle, info)


def hartree_fock(xyz, chkfile, spin=0, basis='vtz'):
    mol = pyscf.gto.M(atom = xyz, basis=f'ccecpccp{basis}', ecp='ccecp', unit='bohr', charge=0, spin=spin, verbose=5)
    mf = pyscf.scf.ROHF(mol)
    mf.callback = partial(save_scf_iteration,chkfile)
    mf.chkfile=chkfile
    dm = mf.init_guess_by_atom()
    mf.kernel(dm)

def unrestricted_hartree_fock(xyz, chkfile, spin=0, basis='vtz'):
    mol = pyscf.gto.M(atom = xyz, basis=f'ccecpccp{basis}', ecp='ccecp', unit='bohr', charge=0, spin=spin)
    mf = pyscf.scf.UHF(mol)
    dm = mf.init_guess_by_atom()
    mf.kernel(dm)

    #check stability
    mo1 = mf.stability()[0]
    rdm1 = mf.make_rdm1(mo1, mf.mo_occ)
    mf.chkfile=chkfile
    mf.callback = partial(save_scf_iteration,chkfile)
    mf = mf.run(rdm1)


def dft(xyz,chkfile, spin=0, basis='vtz', functional = 'pbe,pbe'):
    #HYB_GGA_XC_WB97X, PBE, HSE, M06
    mol = pyscf.gto.M(atom = xyz, basis=f'ccecpccp{basis}', ecp='ccecp', unit='bohr', charge=0, spin=spin)
    mf = pyscf.scf.ROKS(mol)
    mf.xc=functional
    mf.chkfile=chkfile
    mf.callback = partial(save_scf_iteration,chkfile)
    dm = mf.init_guess_by_atom()
    mf.kernel(dm)

def mean_field(xyz,chkfile, functional, settings=None, **kwargs):

    if functional=='hf':
        hartree_fock(xyz,chkfile, spin=settings['spin'], **kwargs)
    elif functional=='uhf':
        unrestricted_hartree_fock(xyz,chkfile, spin=settings['spin'], **kwargs)
    else:
        dft(xyz,chkfile, functional=functional, spin=settings['spin'], **kwargs)


def run_hci(hf_chkfile, chkfile, select_cutoff=0.1, nroots=4):
    mol, mf = pyq.recover_pyscf(hf_chkfile, cancel_outputs=False)
    cisolver = pyscf.hci.SCI(mol)
    cisolver.select_cutoff=select_cutoff
    cisolver.nroots=nroots
    nmo = mf.mo_coeff.shape[1]
    nelec = mol.nelec
    h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
    h2 = pyscf.ao2mo.full(mol, mf.mo_coeff)
    e, civec = cisolver.kernel(h1, h2, nmo, nelec, verbose=0)
    cisolver.ci= np.array(civec)
    rdm1,rdm2 = cisolver.make_rdm12s(civec[0], nmo, nelec)
    pyscf.lib.chkfile.save(chkfile,'ci',
        {'ci':cisolver.ci,
        'nmo':nmo,
        'nelec':nelec,
        '_strs':cisolver._strs,
        'select_cutoff':select_cutoff,
        'energy':e+mol.energy_nuc(),
        'rdm':np.array(rdm1)
        })

def fci(hf_chkfile, fci_chkfile, nroots=4):
    mol, mf = pyq.recover_pyscf(hf_chkfile, cancel_outputs=False)
    cisolver = pyscf.fci.FCI(mf)
    cisolver.nroots = nroots
    cisolver.kernel()
    print(dir(cisolver))
    with h5py.File(fci_chkfile, "w") as f:
        f["e_tot"] = cisolver.e_tot

def recover_hci(hf_chkfile, ci_chkfile):
    mol, mf = pyq.recover_pyscf(hf_chkfile)
    cisolver = pyscf.hci.SCI(mol)
    cisolver.__dict__.update(pyscf.lib.chkfile.load(ci_chkfile,'ci'))
    return mol, mf, cisolver

def generate_wf_gs(hf_chkfile, ci_chkfile, slater_kws=None):
    if ci_chkfile is not None:
        mol, mf, cisolver = recover_hci(hf_chkfile, ci_chkfile)
        cisolver.ci=cisolver.ci[0]
        wf, to_opt = pyq.generate_wf(mol, mf, mc=cisolver, slater_kws=slater_kws)
    else: 
        mol, mf = pyq.recover_pyscf(hf_chkfile)
        wf, to_opt = pyq.generate_wf(mol, mf, slater_kws=slater_kws)
    return mol, mf, wf, to_opt

def optimize_gs(hf_chkfile, ci_chkfile, opt_chkfile, start_from=None, slater_kws=None, nconfig=1000, **kwargs):
    mol, mf, wf, to_opt = generate_wf_gs(hf_chkfile, ci_chkfile, slater_kws)
    if start_from is not None:
        print("reading from", start_from)
        pyq.read_wf(wf, start_from)
    configs = pyq.initial_guess(mol, nconfig)
    acc = pyq.gradient_generator(mol, wf, to_opt)
    pyq.line_minimization(wf, configs, acc, verbose=True, hdf_file = opt_chkfile, **kwargs)

def transform_ci(ci, coeffs):
    return np.sum([ci[i] * coeffs[i] for i in range(len(coeffs))], axis=0)


def generate_wfs(hf_chkfile, cas_chkfile, weights, anchor_wfs, slater_kws=None, jastrow_kws=None):
    mol, mf, cas = recover_hci(hf_chkfile, cas_chkfile)
    cas.ci = transform_ci(cas.ci, weights)
    print(cas.ci)

    wf_anchor = []
    for anchor in anchor_wfs:
        wf, to_opt = pyq.generate_wf(
            mol,
            mf,
            mc=cas,
            slater_kws=slater_kws,
            jastrow_kws=jastrow_kws
        )
        pyq.read_wf(wf, anchor)
        wf_anchor.append(wf)

    wf_es, to_opt = pyq.generate_wf(
        mol,
        mf,
        mc=cas,
        slater_kws=slater_kws,
        jastrow_kws=jastrow_kws
    )

    for k in wf_es.parameters.keys():
        if "wf2" in k:
            wf_es.parameters[k] = wf_anchor[0].parameters[k].copy()
            print(wf_es.parameters[k])
    return mol, wf_anchor, wf_es, to_opt


def orthogonal_opt(
    hf_chkfile, cas_chkfile, anchor_wfs, ortho_chkfile, Starget=None, nconfig=500, start_from=None, slater_kws=None, jastrow_kws=None, **kws
):
    if Starget is None:
        Starget = np.zeros(len(anchor_wfs))

    final_weight = 1.0-np.sum(np.abs(Starget)**2)

    mol, wf_anchor, wf_es, to_opt = generate_wfs(
        hf_chkfile, cas_chkfile, np.append(Starget,[final_weight]), anchor_wfs,
        slater_kws=slater_kws, jastrow_kws=jastrow_kws
    )
    for k in ["wf1det_coeff", "wf1mo_coeff_alpha", "wf1mo_coeff_beta"]:
        if k in to_opt.keys():
            to_opt[k] = np.ones_like(to_opt[k], dtype="bool")
    if start_from is not None:
        pyq.read_wf(wf_es, start_from)
    configs = pyq.initial_guess(mol, nconfig)
    acc = pyq.gradient_generator(mol, wf_es, to_opt)
    pyq.optimize_orthogonal(
        wf_anchor+[wf_es],
        configs,
        acc,
        Starget[0:len(anchor_wfs)],
        hdf_file=ortho_chkfile,
        **kws,
    )


def generate_accumulators(mol, mf):
    if len(mf.mo_coeff.shape)==2:
        mo_coeff = [mf.mo_coeff, mf.mo_coeff]
    else:
        mo_coeff=mf.mo_coeff
    return {
        'energy':pyq.EnergyAccumulator(mol),
        'rdm1_up':pyqmc.obdm.OBDMAccumulator(mol, orb_coeff=mo_coeff[0], spin=0),
        'rdm1_down': pyqmc.obdm.OBDMAccumulator(mol, orb_coeff=mo_coeff[1], spin=1),
        #'rdm2_updown': pyqmc.tbdm.TBDMAccumulator(mol, orb_coeff=mo_coeff2rdm, spin=(0,1)),
        #'rdm2_upup': pyqmc.tbdm.TBDMAccumulator(mol, orb_coeff=mo_coeff2rdm, spin=(0,0))
    }

def evaluate_vmc(hf_chkfile, ci_chkfile, opt_chkfile, vmc_chkfile, slater_kws=None, nconfig=1000, **kwargs):
    mol, mf, wf, to_opt = generate_wf_gs(hf_chkfile, ci_chkfile, slater_kws)
    configs = pyq.initial_guess(mol, nconfig)
    pyq.read_wf(wf, opt_chkfile)
    acc=generate_accumulators(mol, mf)
    pyq.vmc(wf, configs, accumulators=acc, verbose=True, hdf_file = vmc_chkfile, **kwargs)


def evaluate_dmc(hf_chkfile, ci_chkfile, opt_chkfile, dmc_chkfile, slater_kws=None, nconfig=1000, **kwargs):
    mol, mf, wf, to_opt = generate_wf_gs(hf_chkfile, ci_chkfile, slater_kws)
    configs = pyq.initial_guess(mol, nconfig)
    pyq.read_wf(wf, opt_chkfile)
    acc=generate_accumulators(mol, mf)
    pyq.rundmc(wf, configs, accumulators=acc, verbose=True, hdf_file = dmc_chkfile, **kwargs)

def run_ccsd(hf_chkfile, chkfile):
    mol, mf = pyq.recover_pyscf(hf_chkfile,cancel_outputs=False)
    mycc = pyscf.cc.CCSD(mf).run(verbose=0)
    dm1 = mycc.make_rdm1()

    if mol.spin ==0:
        from pyscf.cc import ccsd_t_lambda_slow as ccsd_t_lambda
        from pyscf.cc import ccsd_t_rdm_slow as ccsd_t_rdm
    else:
        from pyscf.cc import uccsd_t_lambda as ccsd_t_lambda
        from pyscf.cc import uccsd_t_rdm as ccsd_t_rdm
    eris = mycc.ao2mo()
    conv, l1, l2 = ccsd_t_lambda.kernel(mycc, eris, mycc.t1, mycc.t2)
    dm1_t = ccsd_t_rdm.make_rdm1(mycc, mycc.t1, mycc.t2, l1, l2, eris=eris)
    pyscf.lib.chkfile.save(chkfile,'ccsd',
        {
        'energy':mycc.e_tot,
        'rdm':dm1
        })
    pyscf.lib.chkfile.save(chkfile,'ccsdt',
        {
        'energy':mycc.ccsd_t(),
        'rdm':dm1_t
        })
