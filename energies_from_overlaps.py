"""Given an HDF5 file with matrix elements and overlaps along with epsilon, 
fill the subspace matrices and compute the eigenvalues as a function of
the subspace dimension d.

The input file shall have the following datasets:
epsilon - a single float with the threshold tolerance.
overlaps - The overlaps <psi_i|U^d|psi_j>.
matrix_elements - The overlaps <psi_i|H U^d|psi_j>.
"""

from typing import Tuple
import h5py
import argparse
import numpy as np
import scipy.linalg as la
from krylov_common import fill_subspace_matrices

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


def krylov_energy(h: np.ndarray, s: np.ndarray) -> float:
    """Get the energy by solving the generalized eigenvalue problem H psi = E S psi."""

    evals, evecs = la.eig(h, s)
    return np.min(evals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="HDF5 file with matrix elements and overlaps, as well as epsilon.")
    parser.add_argument("output_file", type=str, help="HDF5 file for ouptut.")
    args = parser.parse_args()

    in_file = h5py.File(args.input_file, "r")
    eps = in_file["eps"][()]
    overlaps = np.array(in_file["overlaps"][:])
    matrix_elements = np.array(in_file["matrix_elements"][:])
    in_file.close()

    assert overlaps.size == matrix_elements.size

    h, s = fill_subspace_matrices(matrix_elements, overlaps)

    # Take the upper d*d blocks of H and S and compute the energy
    # as a function of d.
    energies = []
    for d in range(overlaps.size):
        h_d = h[:d, :d]
        s_d = s[:d, :d]
        new_h, new_s = threshold_eigenvalues(h_d, s_d, eps)
        energy = krylov_energy(new_h, new_s)
        energies.append(energy)
    
    out_file = h5py.File(args.output_file, "w")
    out_file.create_dataset("eps", data=eps)
    out_file.create_dataset("matrix_elements", data=matrix_elements)
    out_file.create_dataset("overlaps", data=overlaps)
    out_file.create_dataset("energies", data=np.array(energies))
    out_file.close()

if __name__ == "__main__":
    main()