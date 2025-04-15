from typing import List, Tuple
import argparse
import numpy as np
import scipy.linalg as la
import h5py
import pandas as pd

def threshold_eigenvalues(h: np.ndarray, s: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Remove all eigenvalues below a positive threshold eps.
    See Epperly et al. sec. 1.2."""

    # Build a matrix whose columns correspond to the positive eigenvectors of s.
    evals, evecs = la.eigh(s)
    positive_evals = []
    positive_evecs = []
    for i, ev in enumerate(evals):
        assert abs(ev.imag) < 1e-7
        if ev.real > eps:
            positive_evals.append(ev.real)
            positive_evecs.append(evecs[:, i])
    pos_evec_mat = np.vstack(positive_evecs).T
    # Project h and s into this subspace.
    new_s =  pos_evec_mat.conj().T @ s @ pos_evec_mat
    new_h = pos_evec_mat.conj().T @ h @ pos_evec_mat
    return new_h, new_s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="HDF5 file with subsapce matrices.")
    parser.add_argument("output_file", type=str, help="HDF5 file for output.")
    args = parser.parse_args()

    # Load subspace matrices from "hubbard_subspace_matrices.h5"
    f = h5py.File(args.input_file, "r")
    # Get the subspace matrices.
    h = np.array(f.get("h"))
    s = np.array(f.get("s"))
    # Save these values to copy
    tau_input = f["tau"][()]
    steps_input = f["steps"][()]
    f.close()
    # assert la.ishermitian(h)
    # assert la.ishermitian(s)
    # Get ground state energies for various subspace dimensions and thresholds.
    results: List[Tuple[int, float, float]] = []
    for d in range(3, h.shape[0]+1):
        for eps in np.logspace(-12, -5, num=4):
            # Get the top d * d blocks of h and s.
            h_d = h[:d, :d]
            s_d = s[:d, :d]
            # Project onto the thresholded subspace.
            new_h, new_s = threshold_eigenvalues(h_d, s_d, eps=eps)
            evals, evecs = la.eigh(new_h, new_s)
            results.append((d, eps, np.min(evals), new_h.shape[0]))
    # Output to HDF5 file.
    df = pd.DataFrame.from_records(results, columns=["d", "eps", "energy", "num_pos"])
    df.index.name = "i"
    df.to_hdf(args.output_file, key="eigenvalues", index=False)
    df.to_csv("eigenvalues.csv")
    f_out = h5py.File(args.output_file, "a")
    f_out.create_dataset("tau", data=tau_input)
    f_out.create_dataset("steps", data=steps_input)

if __name__ == "__main__":
    main()