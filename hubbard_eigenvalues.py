import numpy as np
import scipy.linalg as la
import h5py

def main():
    # Load subspace matrices from "hubbard_subspace_matrices.h5"
    f = h5py.File("hubbard_subspace_matrices.h5", "r")
    h = np.array(f.get("h"))
    s = np.array(f.get("s"))
    f.close()
    print(la.norm(h - np.conjugate(h).T))
    print(la.norm(s - np.conjugate(s).T))
    # Use eigenvalue thresholding to get the eigenvalues.

if __name__ == "__main__":
    main()