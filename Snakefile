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


rule OPTIMIZE_HCI:
    input: mf = "{dir}/mf.chk", hci="{dir}/hci{hci_tol}.chk"  #unpack(opt_dependency), 
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



        with concurrent.futures.ProcessPoolExecutor(max_workers=qmc_threads) as client:
            pyq.OPTIMIZE(input.mf, 
                        output[0],
                        ci_checkfile=input.hci,
                        load_parameters=None,                      
                        slater_kws=slater_kws, 
                        nconfig = nconfig, 
                        vmcoptions = vmcoptions, 
                        client=client, 
                        npartitions=qmc_threads)
