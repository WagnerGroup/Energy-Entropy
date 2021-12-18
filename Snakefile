import functions
import concurrent
import numpy as np
import pyqmc.api as pyq
qmc_threads=2
partition="wagner"
import json

rule MEAN_FIELD:
    output: "h{natoms}_{bond_length}/{functional}/{basis}/mf.chk"
    resources:
        walltime="4:00:00", partition=partition
    run:
        functions.mean_field(natoms=int(wildcards.natoms), 
                            bond_length=float(wildcards.bond_length),
                            chkfile=output[0],  
                            basis=wildcards.basis, 
                            functional=wildcards.functional)

rule HCI:
    input: "{dir}/mf.chk"
    output: "{dir}/hci{tol}.chk"
    resources:
        walltime="4:00:00", partition=partition
    run:
        functions.run_hci(input[0],output[0], float(wildcards.tol), nroots=1)

rule CC:
    input: "{dir}/mf.chk"
    output: "{dir}/cc.chk"
    threads: qmc_threads
    resources:
        walltime="48:00:00", partition=partition
    run:
        functions.run_ccsd(input[0],output[0])

rule FCI:
    input: "{dir}/mf.chk"
    output: "{dir}/fci.chk"
    run:
        functions.fci(input[0], output[0], nroots=1)


nblocks = [10,20,40,60,80,100]

def opt_dependency(wildcards):
    d = {}
    basedir = f"{wildcards.dir}/"
    ind = nblocks.index(int(wildcards.nblocks))
    if ind > 0:
        if hasattr(wildcards,'hci_tol'):
            basefile = basedir + f"opt_hci{wildcards.hci_tol}_{wildcards.determinant_cutoff}_"
        else:
            basefile = basedir + f"opt_mf_"
        basefile = basefile + f"{nblocks[ind-1]}.chk"
        d["load_parameters"] = basefile
    return d


rule OPTIMIZE_HCI:
    input: unpack(opt_dependency), mf = "{dir}/mf.chk", hci="{dir}/hci{hci_tol}.chk"
    output: "{dir}/opt_hci{hci_tol}_{determinant_cutoff}_{nblocks}.chk"
    threads: qmc_threads
    resources:
        walltime="72:00:00", partition=partition
    run:
        nconfig = 400
        vmcoptions = {'nblocks':int(wildcards.nblocks)}
        slater_kws={'optimize_orbitals':True, 
                    'optimize_zeros':False,
                    'optimize_determinants':True,
                    'tol':float(wildcards.determinant_cutoff)
                  }
        load_parameters = None
        if hasattr(input, 'load_parameters'):
            load_parameters=input.load_parameters
        with concurrent.futures.ProcessPoolExecutor(max_workers=qmc_threads) as client:
            pyq.OPTIMIZE(input.mf, 
                        output[0],
                        ci_checkfile=input.hci,
                        load_parameters = load_parameters,                      
                        slater_kws=slater_kws, 
                        nconfig = nconfig, 
                        vmcoptions = vmcoptions, 
                        client=client, 
                        npartitions=qmc_threads)

rule VMC:
    input: mf = "{dir}/mf.chk", opt = "{dir}/opt_{variables}.chk"
    output: "{dir}/vmc_{variables}.chk"
    threads: qmc_threads
    resources:
        walltime="24:00:00", partition=partition
    run:
        slater_kws = None
        variables = wildcards.variables.split('_')
        startingwf = variables[0]
        if 'hci' in startingwf:
            determinant_cutoff = variables[1]
            slater_kws={'tol':float(determinant_cutoff)}
        with concurrent.futures.ProcessPoolExecutor(max_workers=qmc_threads) as client:
            pyq.VMC(input.mf, 
                    output[0],
                    load_parameters = input.opt, 
                    ci_checkfile = wildcards.dir+"/"+startingwf+".chk",                 
                    slater_kws = slater_kws, 
                    nconfig = 1000, 
                    nblocks = 60, 
                    client=client, 
                    npartitions=qmc_threads)


rule DMC:
    input: mf = "{dir}/mf.chk", opt = "{dir}/opt_{variables}.chk"
    output: "{dir}/dmc_{variables}_{tstep}.chk"
    threads: qmc_threads
    resources:
        walltime="24:00:00", partition=partition
    run:
        slater_kws = None
        variables = wildcards.variables.split('_')
        startingwf = variables[0]
        if 'hci' in startingwf:
            determinant_cutoff = variables[1]
            slater_kws={'tol':float(determinant_cutoff)}

        with concurrent.futures.ProcessPoolExecutor(max_workers=qmc_threads) as client:
            pyq.DMC(input.mf, 
                    output[0],
                    load_parameters = input.opt, 
                    ci_checkfile = wildcards.dir+"/"+startingwf+".chk",                
                    slater_kws=slater_kws, 
                    nconfig = 1000, 
                    tstep=float(wildcards.tstep), 
                    nsteps=int(30/float(wildcards.tstep)),
                    client=client, 
                    npartitions=qmc_threads)
