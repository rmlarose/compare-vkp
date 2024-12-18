"""Some utility functions for OpenFermion."""

import cirq
import openfermion as of


def get_qubits(hamiltonian: of.QubitOperator) -> set[int]:
    qubits = set()
    for p in hamiltonian.get_operators():
        for qubit, _ in list(p.terms.keys())[0]:
            qubits.add(qubit)
    return qubits


def get_num_qubits(hamiltonian: of.QubitOperator) -> int:
    return len(get_qubits(hamiltonian))


def preprocess_hamiltonian(
    hamiltonian: of.QubitOperator,
    drop_term_if = None,
) -> cirq.PauliSum:
    """Preprocess the Hamiltonian and convert it to a cirq.PauliSum."""
    if drop_term_if is None:
        drop_term_if = []

    new = cirq.PauliSum()

    for term in hamiltonian.terms:
        add_term = True

        for drop_term in drop_term_if:
            if drop_term(term):
                add_term = False
                break

        if add_term:
            key = " ".join(pauli + str(index) for index, pauli in term)
            new += next(iter(
                of.transforms.qubit_operator_to_pauli_sum(
                    of.QubitOperator(key, hamiltonian.terms.get(term)
                )
            )))

    return new
