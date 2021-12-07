import h5py
import glob
import numpy as np

for fname in glob.glob("*/*/*/hci*.chk"):
    with h5py.File(fname) as f:
        dm = f['/ci/rdm'][()]
        w, v = np.linalg.eigh(dm)
        w = w[np.abs(w)>0]
        entropy=-np.sum(w*np.log(w))
        print(fname, f['/ci/energy'][()], entropy)