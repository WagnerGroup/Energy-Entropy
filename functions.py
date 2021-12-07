import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pyscf
import h5py
import pyqmc.tbdm
import pyscf.hci
import pyqmc.api as pyq
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


def mean_field(natoms, bond_length,chkfile, functional, **kwargs):
    xyz = ";".join(f"H 0. 0. {i*bond_length}" for i in range(natoms))
    if functional=='hf':
        hartree_fock(xyz,chkfile, **kwargs)
    elif functional=='uhf':
        unrestricted_hartree_fock(xyz,chkfile, **kwargs)


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
    with h5py.File(fci_chkfile, "w") as f:
        f["e_tot"] = cisolver.e_tot

def recover_hci(hf_chkfile, ci_chkfile):
    mol, mf = pyqmc.recover_pyscf(hf_chkfile)
    cisolver = pyscf.hci.SCI(mol)
    cisolver.__dict__.update(pyscf.lib.chkfile.load(ci_chkfile,'ci'))
    return mol, mf, cisolver


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
