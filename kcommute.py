"""Code for the k-commuting method from https://arxiv.org/abs/2312.11840."""

import math
from typing import Iterable, List

import cirq
import numpy as np
import openfermion as of


def compute_blocks(qubits: Iterable[cirq.Qid], k: int) -> List[List[cirq.Qid]]:
    return [qubits[k * i : k * (i + 1)] for i in range(math.ceil(len(qubits) / k))]


def restrict_to(
    pauli: cirq.PauliString, qubits: Iterable[cirq.Qid]
) -> cirq.PauliString:
    """Returns the Pauli string restricted to the provided qubits.

    Args:
        pauli: A Pauli string.
        qubits: A set of qubits.

    Returns:
        The provided Pauli string acting only on the provided qubits.
        Note: This could potentially be empty (identity).
    """
    return cirq.PauliString(p.on(q) for q, p in pauli.items() if q in qubits)


def commutes(pauli1: cirq.PauliString, pauli2: cirq.PauliString, blocks: List[List[cirq.Qid]]) -> bool:
    """Returns True if pauli1 k-commutes with pauli2, else False.

    Args:
        pauli1: A Pauli string.
        pauli2: A Pauli string.
        blocks: The block partitioning.
    """
    for block in blocks:
        if not cirq.commutes(restrict_to(pauli1, block), restrict_to(pauli2, block)):
            return False
    return True


def get_terms_ordered_by_abscoeff(ham: cirq.PauliSum) -> List[cirq.PauliString]:
    """Returns the terms of the PauliSum ordered by coefficient absolute value.

    Args:
        ham: A PauliSum.

    Returns:
        a list of PauliStrings sorted by the absolute value of their coefficient.
    """
    return sorted([term for term in ham], key=lambda x: abs(x.coefficient), reverse=True)


def get_si_sets(ham: cirq.PauliSum, k: int = 1) -> List[List[cirq.PauliString]]:
    """Returns grouping from the sorted insertion algorithm [https://quantum-journal.org/papers/q-2021-01-20-385/].

    Args:
        ham: The observable to group.
        k: The integer k in k-commutativity.
    """
    qubits = sorted(set(ham.qubits))
    blocks = compute_blocks(qubits, k)

    commuting_sets = []
    terms = get_terms_ordered_by_abscoeff(ham)
    for pstring in terms:
        found_commuting_set = False

        for commset in commuting_sets:
            cant_add = False

            for pauli in commset:
                if not commutes(pstring, pauli, blocks):
                    cant_add = True
                    break

            if not cant_add:
                commset.append(pstring)
                found_commuting_set = True
                break

        if not found_commuting_set:
            commuting_sets.append([pstring])

    return commuting_sets


def get_variance(qubop: of.QubitOperator, psi: np.ndarray):
    """Returns the variance ⟨psi|O^2|psi⟩ - ⟨psi|O|psi⟩^2 where O is the QubitOperator `qubop`."""
    op = of.linalg.get_sparse_operator(qubop, n_qubits=int(np.log2(len(psi))))
    opsq = op @ op

    mean = psi.conj().T @ (op @ psi)
    opsq_expect = psi.conj().T @ (opsq @ psi)
    return opsq_expect - mean ** 2


def compute_shots(groups: List[of.QubitOperator], psi: np.ndarray, epsilon: float) -> int:
    """Returns the shots required to compute the expectation value of the
    grouped operator with respect to the state psi to an accuracy epsilon.

    Args:
        groups: The grouping of the operator to compute the expectation value of. 
        psi: The state to compute the expectation value with respect to.
        epsilon: The desired error.
    """
    temp = 0.0
    for op in groups:
        var = get_variance(op, psi)
        temp += np.sqrt(var)

    temp = temp**2
    shotcounts = temp.real / epsilon ** 2

    return int(round(shotcounts.real))
