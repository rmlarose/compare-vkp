from typing import List, Tuple
import argparse
import json
import h5py
import numpy as np
import scipy.linalg as la
import pandas as pd

def threshold_eigenvalues(h: np.ndarray, s: np.ndarray, eps: float, verbose: bool=False) -> Tuple[np.ndarray, np.ndarray]:
    """Remove all eigenvalues below a positive threshold eps.
    See Epperly et al. sec. 1.2."""

    # Build a matrix whose columns correspond to the positive eigenvectors of s.
    evals, evecs = la.eigh(s)
    if verbose:
        print("All eigenvalues of S:", evals)
    positive_evals = []
    positive_evecs = []
    num_kept = 0
    for i, ev in enumerate(evals):
        assert abs(ev.imag) < 1e-7
        if ev.real > eps:
            positive_evals.append(ev.real)
            positive_evecs.append(evecs[:, i])
            num_kept += 1
    if verbose:
        print(f"Kept {num_kept} eigenvalues out of {len(evals)}.")
    pos_evec_mat = np.vstack(positive_evecs).T
    # Project h and s into this subspace.
    new_s =  pos_evec_mat.conj().T @ s @ pos_evec_mat
    new_h = pos_evec_mat.conj().T @ h @ pos_evec_mat
    return new_h, new_s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="JSON file with parameters.")
    parser.add_argument("subspace_input_file", type=str, help="HDF5 file with subsapce matrices.")
    parser.add_argument("output_file", type=str, help="HDF5 file for output.")
    args = parser.parse_args()

    with open(args.input_file) as f:
        input_dict = json.load(f)
    d = input_dict["d"]
    eps = input_dict["eps"]


    # Load subspace matrices from "hubbard_subspace_matrices.h5"
    f = h5py.File(args.subspace_input_file, "r")
    # Get the subspace matrices.
    h = np.array(f.get("h"))
    s = np.array(f.get("s"))
    # Save these values to copy
    # tau_input = f["tau"][()]
    # steps_input = f["steps"][()]
    f.close()
    # assert la.ishermitian(h)
    # assert la.ishermitian(s)
    # Get ground state energies for various subspace dimensions and thresholds.
    results: List[Tuple[int, float, float, float]] = []
    for d in range(3, h.shape[0]+1):
        # Get the top d * d blocks of h and s.
        h_d = h[:d, :d]
        s_d = s[:d, :d]
        # Project onto the thresholded subspace.
        new_h, new_s = threshold_eigenvalues(h_d, s_d, eps=eps)
        evals, evecs = la.eigh(new_h, new_s)
        print(d, np.min(evals))
        results.append((d, eps, np.min(evals), new_h.shape[0]))
    print("Final result:", results[-1])

    # Output to HDF5 file.
    f_out = h5py.File(args.output_file, "w")
    # f_out.create_dataset("tau", data=tau_input)
    # f_out.create_dataset("steps", data=steps_input)
    f_out.create_dataset("eps", data=eps)
    f_out.close()
    df = pd.DataFrame.from_records(results, columns=["d", "eps", "energy", "num_pos"])
    df.index.name = "i"
    df.to_hdf(args.output_file, key="eigenvalues", index=False)
    df.to_csv("eigenvalues.csv")

if __name__ == "__main__":
    main()
