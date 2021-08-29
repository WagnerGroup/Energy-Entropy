import numpy as np
import itertools
import os
import glob


def geom(n,dist):
    string=f'H 0. 0. 0.0'+'\n'
    for i in range(1,n):
        length="{:.1f}".format(dist*i)
        string+=f'H 0.0 0.0 {length}'+'\n'
    return string

def make_paths(system):
    l = system[1]
    sys = system[0]
    folder = f'{sys}_{l}'
    if os.path.exists(folder):
        if os.path.exists(f'{folder}/geom.xyz'):
            return
    else: 
        os.mkdir(folder)
    s = geom(int(sys[1:]), l)
    f = open(f"{folder}/geom.xyz","w")
    f.write(s)
    f.close
    return 


if __name__=="__main__":

    molecule = ['h2','h4','h6'] #,'h8','h10'
    bond_lengths = [1.4,2.0,3.0] #

    folder = itertools.product(molecule,bond_lengths)
    for molecule,length in folder:
        make_paths([molecule,length])
