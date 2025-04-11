import argparse
from typing import List
from math import sqrt
import pickle
import h5py
import pandas as pd
import numpy as np
from scipy.constants import pi
import scipy.linalg as la
import cirq
import openfermion as of

def krylov_energy_bound(
    d: int,
    norm_h: float,
    chi: float,
    zeta: float,
    gamma0_sq: float,
    delta: float
) -> float:
    """Error bound from Kirby's thm.

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
        2 * chi / delta_p + zeta + 8.0 * (1 + (pi * delta_p) / (4 * norm_h) ** (-2.0 * d))
    )


def chi_for_groups(
    d: int,
    r: int,
    tau: float,
    h_norm: float,
    groups: List[List[cirq.PauliString]]
) -> float:
    """Get a bound on chi.
    
    Arguments:
    d - Subspace dimension.
    r - number of trotter steps used in U.
    tau - total evolution time for U.
    groups - groupings produced by get_si_sets.

    Returns:
    Upper bound on chi."""

    # Turn each group of strings in to a PauliSum.
    psums = [sum(group) for group in groups]
    # Then evaluate sum_gamma1 ||[\sum_gamma2 H_gamma2, H_gamma1]||
    comm_norms = 0.0 # = sum_gamma1 ||\sum_gamma2 [H_gamma2, H_gamma1]||
    for i1 in range(len(psums)):
        total_commutator = cirq.PauliSum() # = \sum_gamma2 [H_gamma2, H_gamma1]
        for i2 in range(i1 + 1, len(psums)):
            total_commutator += psums[i2] * psums[i1] - psums[i1] * psums[i2]
        comm_norms += la.norm(total_commutator.matrix())
    return 2 * h_norm * sqrt(2 * d**4 * r**2 * (tau**2 / (2.0 * r**2) * comm_norms)**2)


def main():
    ham_fermi = of.hamiltonians.fermi_hubbard(2, 2, 1.0, 2.0, spinless=True)
    ham: of.QubitOperator = of.transforms.jordan_wigner(ham_fermi)
    norm_h = la.norm(of.linalg.qubit_operator_sparse(ham))
    exact_file = h5py.File("hubbard_exact.h5", "r")
    ground_state = exact_file["eigenvectors"][:, 0]
    ref_state = exact_file["ref_state"][:]
    # TODO gamma0_sq is small. Better reference state needed?
    gamma0_sq = abs(np.vdot(ref_state, ground_state)) ** 2
    delta = abs(exact_file["energies"][1] - exact_file["energies"][0])
    d = 16
    r = 100
    tau = 0.1
    eps = 1e-12
    zeta = 4 * d * eps
    with open("hubbard_groups.pkl", "rb") as f:
        groups = pickle.load(f)
    chi = chi_for_groups(d, r, tau, norm_h, groups)
    bound = krylov_energy_bound(d, norm_h, chi, zeta, gamma0_sq, delta)
    print(f"bound = {bound}")

if __name__ == "__main__":
    main()