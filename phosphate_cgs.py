"""The Harrow QPE paper says that the error in their QPE algorithm scales as
C_gs delta^p, where C_gs is some constant that can be derived numerically for each molecule.
The show plots of C_gs vs. alpha (a commutator sum) and lambda (the qDRIFT normalization).
Let's compute alpha and lambda for the phosphate."""

from typing import Tuple
import cirq
import openfermion as of

def pauli_string_commutator_norm(p1, p2) -> float:
    """Given two Pauli strings P1 and P2, return ||[P1, P2]||.
    P1 and P2 are given in openfermion form, e.g.
    IXIZY -> ((1, 'X), (3, 'Z), (4, 'Y'))."""

    # To determine if the strings commute or anticommute, find the set
    # of common indices. For each qubit, count how many sites have non-matching,
    # non-identity Paulis. If this number is even, then the strings commute.
    # If this number is odd, then the strings anti-commute.
    # e.g. XX and ZZ have non-idenity, non-matching Paulis on both qubits.
    # These strings commute.
    # XY and ZY have non-idenity, non-matching Paulis on only one qubit, so
    # they anticommute.
    idx1 = set([i for _, i in p1])
    idx2 = set([i for _, i in p2])
    indices = sorted(list(idx1 | idx2))

    num_non_matching = 0
    for i in indices:
        if i in idx1 and i in idx2:
            if dict(p1)[i] != dict(p2)[i]:
                num_non_matching += 1
    if num_non_matching % 2 == 0:
        # The strings commute. ||[P1, P2]|| = 0.
        return 0.
    else:
        # The strings anti-commute. ||[P1, P2]|| = 2 ||P1 P2|| = 2.
        return 2.


def main():
    hamiltonian = of.utils.load_operator(file_name="owp_631gd_22_ducc.data", data_directory=".")
    hamiltonian_qubop = of.transforms.jordan_wigner(hamiltonian)
    nterms = len(hamiltonian_qubop.terms)
    nqubits = of.utils.count_qubits(hamiltonian_qubop)
    print(f"Hamiltonian as {nterms} terms and {nqubits} qubits.")

    alpha = 0.
    term_list = list(hamiltonian_qubop.terms.keys())
    for i in range(nterms):
        if i % 100 == 0:
            print(f"i = {i}")
        h1 = hamiltonian_qubop.terms[term_list[i]]
        p1 = term_list[i]
        for j in range(i+1, nterms):
            h2 = hamiltonian_qubop.terms[term_list[j]]
            p2 = term_list[j]
            comm_norm = pauli_string_commutator_norm(p1, p2)
            alpha += abs(h1 * h2) * comm_norm
    print(f"alpha = {alpha:4.5e}")

if __name__ == "__main__":
    main()