import argparse
from typing import List
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
    delta: float
) -> float:
    """Error bound from Kirby's thm. See Eq. 50.

    Arguments:
    d - subspace dimension.
    norm_h - Norm of Hamiltonian matrix (in full basis, not subspace).
    chi - chi from Kirby's equation.
    zeta - zeta from Kirby's equation.
    gamma0_sq - |<reference | ground state >|^2.
    delta - energy gap of the Hamiltonian.

    Returns:
    upper bound on error in ground state energy."""

    delta_p = delta - chi / gamma0_sq
    return chi / gamma0_sq + (6 * norm_h) / gamma0_sq * (
        2 * chi / delta_p + zeta + 8.0 * (1 + (pi * delta_p) / (4 * norm_h)) ** (-2.0 * d)
    )


def chi_for_groups(
    d: int,
    r: int,
    tau: float,
    h_norm: float,
    comm_norms: float
) -> float:
    """Get a bound on chi.
    
    Arguments:
    d - Subspace dimension.
    r - number of trotter steps used in U.
    tau - total evolution time for U.
    comm_norms - sum of commutator norms from Prop. 9 of ToTECS.

    Returns:
    Upper bound on chi."""

    return 2 * h_norm * (d ** 2 * tau ** 2) / (sqrt(2.0) * r) * comm_norms


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


def main():
    # Load Hamiltonian.
    ham: of.QubitOperator = load_hubbard_hamiltonian()
    ham_cirq = of.qubit_operator_to_pauli_sum(ham)
    nq = len(ham_cirq.qubits)
    norm_h = norm(of.linalg.qubit_operator_sparse(ham), ord=2)

    # Get data from FCI calculation.
    exact_file = h5py.File("hubbard_exact.h5", "r")
    ground_state = exact_file["eigenvectors"][:, 0]
    ref_state = exact_file["ref_state"][:]
    #gamma0_sq = abs(np.vdot(ref_state, ground_state)) ** 2
    gamma0_sq = (1.0 - 1e-2) ** 2
    print(f"gamma0^2 = {gamma0_sq}")
    delta = abs(exact_file["energies"][1] - exact_file["energies"][0])
    print(f"Delta = {delta}")

    # Compute bound.
    d = 16
    r = 100
    tau = 0.1
    eps = 1e-12
    groups = get_si_sets(ham_cirq, nq)
    groups_of = to_groups_of(groups)
    comm_norm = first_order_comm_norm(groups_of, nq)
    s_dist = d ** 2 * tau ** 2 / (2.0 * r) * comm_norm # Bound on ||S - S'||.
    print(f"||S-S'|| <= {s_dist:.4f}")
    zeta = 2 * d * (eps + s_dist)
    print(f"zeta = {zeta}")
    chi = chi_for_groups(d, r, tau, norm_h, comm_norm)
    print(f"chi = {chi}")
    bound = krylov_energy_bound(d, norm_h, chi, zeta, gamma0_sq, delta)
    print(f"bound = {bound}")

if __name__ == "__main__":
    main()