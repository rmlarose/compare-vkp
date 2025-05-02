from typing import List, Tuple
import argparse
from math import sqrt
import pickle
import h5py
import pandas as pd
import numpy as np
from scipy.constants import pi
from scipy.sparse.linalg import norm
import cirq
import openfermion as of
from convert import to_groups_of
from krylov_common import load_hubbard_hamiltonian
from kcommute import get_si_sets

def krylov_energy_bound(
    d: int,
    norm_h: float,
    chi: float,
    zeta: float,
    gamma0_sq: float,
    delta: float,
    eps: float,
    s_dist: float
) -> float:
    """Error bound from Kirby's thm. See Eq. 50.

    Arguments:
    d - subspace dimension.
    norm_h - Norm of Hamiltonian matrix (in full basis, not subspace).
    chi - chi from Kirby's equation.
    zeta - zeta from Kirby's equation.
    gamma0_sq - |<reference | ground state>|^2.
    delta - energy gap of the Hamiltonian.

    Returns:
    upper bound on error in ground state energy."""

    delta_p = delta - chi / gamma0_sq
    gamma0_sq_p = gamma0_sq - 2.0 * eps - 2 * s_dist
    t1 = 2 * chi / delta_p
    t2 = zeta
    t3 = 8.0 * (1 + (pi * delta_p) / (4 * norm_h)) ** (-2.0 * d)
    p1 = chi / gamma0_sq_p
    p2 = (6 * norm_h) / gamma0_sq_p
    return p1 + p2 * (t1 + t2 + t3)


def first_order_comm_norm(
    groups: List[of.QubitOperator],
    nq: int
) -> float:
    """Norm of commutators from first-order Trotter bound in ToTECS Proposition 9.

    Arguments:
    groups: A list of qubit operators. Each represents a term in the Trotter expansion.

    Returns:
    Sum of commutator norms from Prop. 9."""

    # Evaluate sum_gamma1 ||[\sum_gamma2 H_gamma2, H_gamma1]||
    comm_norms = 0.0 # = sum_gamma1 ||\sum_gamma2 [H_gamma2, H_gamma1]||
    for i1 in range(len(groups)):
        total_commutator = of.QubitOperator() # = \sum_gamma2 [H_gamma2, H_gamma1]
        for i2 in range(i1 + 1, len(groups)):
            total_commutator += groups[i2] * groups[i1] - groups[i1] * groups[i2]
        tc_matrix = of.linalg.qubit_operator_sparse(total_commutator, nq)
        comm_norms += norm(tc_matrix, ord=2)
    return comm_norms


def s_dist_bound(d: int, tau: float, r: int, comm_norm: float) -> float:
    """Upper bound on ||S-S'|| from Trotter error.
    
    Arguments:
    d - subspace dimension.
    tau - evolution time
    r - number of time steps.
    comm_norm - sum of commutator norms.
    
    Returns:
    upper bound on ||S-S'||."""

    return d ** 2 * tau ** 2 / (sqrt(2.0) * r) * comm_norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exact_file", type=str, help="File with FCI calculation.")
    parser.add_argument("subspace_file", type=str, help="File with subspace matrix calculation.")
    parser.add_argument("eigenvalue_file", type=str, help="File with eigenvalue calculations.")
    parser.add_argument("output_file", type=str, help="File for putting out bound.")
    args = parser.parse_args()

    print("Computing energy bound.")

    # Load Hamiltonian.
    ham: of.QubitOperator = load_hubbard_hamiltonian()
    ham_cirq = of.qubit_operator_to_pauli_sum(ham)
    nq = len(ham_cirq.qubits)
    norm_h = norm(of.linalg.qubit_operator_sparse(ham), ord=2)

    # Get data from FCI calculation.
    exact_file = h5py.File(args.exact_file, "r")
    ground_state = exact_file["eigenvectors"][:, 0]
    delta = abs(exact_file["energies"][1] - exact_file["energies"][0])
    exact_file.close()

    # Get data from subspace matrix calculation.
    subspace_file = h5py.File(args.subspace_file, "r")
    ref_state = subspace_file["ref_state"][:]
    tau = subspace_file["tau"][()]
    r = subspace_file["steps"][()]
    d_max = np.array(subspace_file["h"]).shape[0]
    gamma0_sq = abs(np.vdot(ref_state, ground_state)) ** 2
    subspace_file.close()

    # Get data from eigenvalue calculation file.
    ev_file = h5py.File(args.eigenvalue_file, "r")
    eps = ev_file["eps"][()]
    ev_file.close()

    # Compute bound at different values of d.
    groups = get_si_sets(ham_cirq, nq)
    groups_of = to_groups_of(groups)
    comm_norm = first_order_comm_norm(groups_of, nq)
    results: List[Tuple[int, float]] = []
    for d in range(d_max):
        s_dist = s_dist_bound(d, tau, r, comm_norm)
        print(f"s_dist={s_dist}")
        zeta = 2 * d * (eps + s_dist)
        chi = 2 * norm_h * s_dist
        bound = krylov_energy_bound(d, norm_h, chi, zeta, gamma0_sq, delta, eps, s_dist)
        results.append((d, bound))
        print(f"d={d}, bound={bound}")

    # Output to file.
    df = pd.DataFrame.from_records(results, columns=["d", "bound"])
    df.set_index("d", inplace=True)
    df.to_hdf(args.output_file, key="bound")

if __name__ == "__main__":
    main()
