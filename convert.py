"""Code to convert between libraries, e.g. from a Cirq.PauliSum to a qiskit.SparsePauliOp."""

from typing import List

import cirq
from cirq.contrib.qasm_import import circuit_from_qasm
import numpy as np
import openfermion as of
import qiskit
from qiskit.quantum_info import SparsePauliOp


def cirq_pauli_sum_to_qiskit_pauli_op(pauli_sum: cirq.PauliSum) -> SparsePauliOp:
    """Returns a qiskit.SparsePauliOp representation of the cirq.PauliSum."""
    cirq_pauli_to_str = {cirq.X: "X", cirq.Y: "Y", cirq.Z: "Z"}

    qubits = pauli_sum.qubits
    terms = []
    coeffs = []
    for term in pauli_sum:
        string = ""
        for qubit in qubits:
            if qubit not in term:
                string += "I"
            else:
                string += cirq_pauli_to_str[term[qubit]]
        terms.append(string)
        assert np.isclose(term.coefficient.imag, 0.0, atol=1e-7)
        coeffs.append(term.coefficient.real)
    return SparsePauliOp(terms, coeffs)


def to_groups_of(groups: List[List[cirq.PauliString]]) -> List[of.QubitOperator]:
    """Convert groups from List[List[cirq.PauliString]] to List[of.QubitOperator]."""
    maps = {cirq.X: "X", cirq.Y: "Y", cirq.Z: "Z"}

    groups_of: List[of.QubitOperator] = []
    for group in groups:
        group_of = of.QubitOperator()

        for p in group:
            group_of += of.QubitOperator(" ".join([f"{maps[v]}{k.x}" for k, v in p._qubit_pauli_map.items()]), coefficient=p.coefficient)
        groups_of.append(group_of)

    return groups_of


def cirq_circuit_to_qiskit(cirq_circuit: cirq.Circuit) -> qiskit.QuantumCircuit:
    """Convert from a cirq circuit to a qiskit circuit by way of QASM."""

    qasm_str = cirq_circuit.to_qasm()
    return qiskit.QuantumCircuit.from_qasm_str(qasm_str)


def qiskit_circuit_to_cirq(qiskit_circuit: qiskit.QuantumCircuit) -> cirq.Circuit:
    """Convert a circuit from qiskit to cirq."""

    qasm_str = qiskit.qasm3.dumps(qiskit_circuit)
    return circuit_from_qasm(qasm_str)
