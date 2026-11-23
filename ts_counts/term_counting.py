"""Count the terms in the Hamiltonians."""

import numpy as np
from openfermion import InteractionOperator, get_fermion_operator, jordan_wigner
from openfermion.utils import count_qubits
from openfermion.ops.representations import get_tensors_from_integrals

def count_qubits_terms(hamiltonian_filename: str) -> int:
    h_data = np.load("../hamiltonians/ts.npz")
    ecore = float(h_data["ECORE"])  # stored as a 0-d array; OpenFermion needs a scalar
    h1 = h_data["H1"]
    # H2 is in chemist notation, (pq|rs). get_tensors_from_integrals expects
    # OpenFermion's MolecularData ordering, two_body_integrals[p,q,r,s] = (ps|qr).
    h2 = h_data["H2"].transpose(0, 2, 3, 1)
    one_body_tensor, two_body_tensor = get_tensors_from_integrals(h1, h2)
    hamiltonian_op = InteractionOperator(
        constant=ecore,
        one_body_tensor=one_body_tensor,
        two_body_tensor=two_body_tensor
    )
    fermion_hamiltonian = get_fermion_operator(hamiltonian_op)
    qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)
    return count_qubits(qubit_hamiltonian), len(qubit_hamiltonian.terms)


def main():
    filenames = [
        "../hamiltonians/ts.npz",
        "../hamiltonians/po3_1w_product_22o.npz",
        "../hamiltonians/po3_1w_reactant_22o.npz",
    ]

    for fname in filenames:
        nq, nterms = count_qubits_terms(fname)
        print(fname, nq, nterms)


if __name__ == "__main__":
    main()