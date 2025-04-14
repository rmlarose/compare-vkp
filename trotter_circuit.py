from typing import List, Optional, Collection, Set
import numpy as np
import cirq
import diagonalize
from kcommute import get_si_sets

def group_qubits(group: List[cirq.PauliString]) -> Set[cirq.Qid]:
    """Get the support of a group of Pauli strings."""

    qs = set()
    for ps in group:
        qs |= set(ps.qubits)
    return qs

def diagonal_pstring_exponential_circuit(pstring: cirq.PauliString, theta: float) -> cirq.Circuit:
    """Implements exp(-i theta c P), where theta is a real number and P a diagonal Pauli 
    string (with coefficient c). See the Paulihedral paper https://arxiv.org/abs/2109.03371."""

    # Get the list of qubits on which P acts like Z.
    support: List[cirq.Qid] = []
    for qid, pauli in pstring.items():
        assert pauli == cirq.I or pauli == cirq.Z, \
            f"Pauli must be diagonal, but got Pauli {pauli}"
        if pauli == cirq.Z:
            support.append(qid)
    # Convert to a circuit for exp(-i theta c_j P_j).
    circuit = cirq.Circuit()
    if len(support) > 0:
        if len(support) > 1:
            for q in reversed(support[1:]):
                circuit.append(cirq.CX(q, support[0]))
        assert abs(pstring.coefficient.imag) <= 1e-7, \
            f"Pauli string {pstring} has coefficient with imaginary part {pstring.coefficient.imag}."
        circuit.append(cirq.rz(2.0 * theta * pstring.coefficient.real).on(support[0]))
        if len(support) > 1:
            for q in reversed(support[1:]):
                circuit.append(cirq.CX(q, support[0]))
    return circuit


def diagonal_group_exp_circuit(pstrings: List[cirq.PauliString], theta: float) -> cirq.Circuit:
    """Get the circuit that exponentiates a group of diagonal Pauli strings,
    i.e. exp(-i A theta), where A = sum_i c_i P_i, and P_i is in {I, Z}^n."""

    term_circuits: List[cirq.Circuit] = []
    for ps in pstrings:
        term_circuits.append(diagonal_pstring_exponential_circuit(ps, theta))
    total_circuit = cirq.Circuit()
    for term_circuit in term_circuits:
        total_circuit += term_circuit
    return total_circuit


def commuting_group_exponential_circuit(
    pstrings: List[cirq.PauliString], theta: float,
    qs: Optional[Collection[cirq.Qid]]=None
) -> cirq.Circuit:
    """Get the exponential circuit representing exp(-i A theta), where A is a PauliSum
    of mutually commuting Pauli strings, and theta is a real parameter."""

    if qs is not None:
        assert group_qubits(pstrings).issubset(set(qs))

    # Get all qubits the strings act on.
    if qs is None:
        qs = set()
        for ps in pstrings:
            qs = qs | set(ps.qubits)
    # Get the diagonalizing circuit U_d and the diagonalized Pauli strings (without coefficients).
    diag_circuit, diagonalized_strings = diagonalize.diagonalize_pauli_strings(pstrings, qs)
    # Construct the circuit U_exp(theta) that exponentiates the diagonalized strings
    exp_circuit = diagonal_group_exp_circuit(diagonalized_strings, theta)
    # Return U_d U_exp(theta) U_d^\dagger, the exponential of the commuting group.
    diag_circuit_inverse = cirq.inverse(diag_circuit)
    return diag_circuit + exp_circuit + diag_circuit_inverse


def first_order_trotter_for_grouping(
    groups: List[List[cirq.PauliString]],
    dt: float, qs: Optional[Collection[cirq.Qid]] = None
) -> cirq.Circuit:
    """Given Pauli string sorted into fully-commuting groups, get a circuit for a Trotter
    step of time dt."""

    group_circuits: List[cirq.Circuit] = []
    for group in groups:
        group_circuits.append(commuting_group_exponential_circuit(group, dt, qs))
    total_circuit = cirq.Circuit()
    for group_circuit in group_circuits:
        total_circuit += group_circuit
    return total_circuit


def first_order_trotter_for_paulisum(psum: cirq.PauliSum, dt: float) -> cirq.Circuit:
    """Get the first-order Trotter circuit for a PauliSum by sorting into fully-commuting groups.
    For each group, we get a diagonalizing circuit. Then we exponentiate the diagonalized groups.

    Arguments:
    psum - PauliSum representing the Hamiltonian.
    dt - float, duration of the time step.

    Returns:
    circuit - Circuit representing a single Trotter step."""

    groups = get_si_sets(psum)
    return first_order_trotter_for_grouping(groups, dt)


def trotter_multistep_from_groups(
    groups: List[List[cirq.PauliString]],
    qs: List[cirq.Qid],
    tau: float,
    steps: int
) -> cirq.Circuit:
    """Simulate total time tau with a number of first-order steps,
    each of length tau / steps."""

    assert steps >= 1
    dt = tau / float(steps)
    step_circuit = first_order_trotter_for_grouping(groups, dt, qs)
    total_circuit = cirq.Circuit()
    for _ in range(steps):
        total_circuit += step_circuit
    return total_circuit
