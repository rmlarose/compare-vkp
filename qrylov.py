from typing import List, Tuple
import numpy as np
import scipy as sp
from scipy.sparse.linalg import expm

def fill_h_and_s_matrices(
    vectors: List[np.ndarray],
    matrix: np.ndarray,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    dim = len(vectors)
    h = np.zeros((dim, dim), dtype=np.complex128)
    s = np.zeros((dim, dim), dtype=np.complex128)

    for i in range(dim):
        for j in range(i, dim):
            if verbose:
                print(i, j)
            hij = vectors[i].conj().T @ matrix @ vectors[j]
            h[i, j] = hij

            if i != j:
                h[j, i] = np.conjugate(hij)

            sij = vectors[i].conj().T @ vectors[j]
            s[i, j] = sij
            if i != j:
                s[j, i] = np.conjugate(sij)
    return h, s


# Based on https://quantum.cloud.ibm.com/docs/en/tutorials/krylov-quantum-diagonalization.
# and Algorithm 1.1 of https://arxiv.org/abs/2110.07492.
def solve_regularized_gen_eig(
    h: np.ndarray,
    s: np.ndarray,
    threshold: float,
) -> float:
    if np.isclose(threshold, 0, atol=1e-10):
        h_reg = h
        s_reg = s
    else:
        s_vals, s_vecs = sp.linalg.eigh(s)
        s_vecs = s_vecs.T
        good_vecs = np.array(
            [vec for val, vec in zip(s_vals, s_vecs) if val > threshold]
        )
        h_reg = good_vecs.conj() @ h @ good_vecs.T
        s_reg = good_vecs.conj() @ s @ good_vecs.T
    return sp.linalg.eigh(h_reg, s_reg)[0][0]


def energy_vs_d(h, s, eps):
    energies = []
    ds = []
    for d in range(1, h.shape[0]):
        krylov_energy = solve_regularized_gen_eig(h[:d, :d], s[:d, :d], threshold=eps)
        energies.append(krylov_energy)
        ds.append(d)
    return ds, energies


def generate_u_subspace(matrix, bvec, dt, subspace_dimension) -> List[np.ndarray]:
    vectors_u = []
    for k in range(-subspace_dimension // 2, subspace_dimension // 2 + 1, 1):
        print(f"k = {k}")
        Uk = expm(-1j * matrix * k * dt)
        vectors_u.append(Uk @ bvec)
    return vectors_u