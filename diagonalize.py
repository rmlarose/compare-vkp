from typing import List, Tuple
from itertools import product
import numpy as np
import cirq

from mitiq import PauliString

def get_stabilizer_matrix_from_paulis(stabilizers, qubits):
    numq = len(qubits)
    nump = len(stabilizers)
    stabilizer_matrix = np.zeros((2*numq, nump))

    cirq.Z._name

    for i, paulistring in enumerate(stabilizers):
        for key, value in paulistring.items():
            if value._name == "X":
                stabilizer_matrix[int(key) + numq, i] = 1
            elif value._name == "Y":
                stabilizer_matrix[int(key), i] = 1
                stabilizer_matrix[int(key) + numq, i] = 1
            elif value._name == "Z":
                stabilizer_matrix[int(key), i] = 1

    return stabilizer_matrix


def get_stabilizer_matrix_signs(stabilizer_matrix):
    """If an input string contains Y, then we represent it by
    XZ = -Y. This function returns (-1)^k for each stabilizer,
    where k is the number of Y's in the original string.
    
    Arguments:
    stabilizer-matrix - a (2 numq) * nump matrix of the tableau.
    
    Returns:
    signs - List of booleans, where False means the sign is +1, and True
    means the sign is -1, i.e. we do (-1)^b = (-1)^(k mod 2)"""

    numq = len(stabilizer_matrix) // 2
    nump = len(stabilizer_matrix[0])
    
    signs: List[bool] = []
    for i in range(nump):
        p = stabilizer_matrix[:, i]
        zs = p[:numq]
        xs = p[numq:]
        k = 0 # Number of Y's in this string.
        for z, x in zip(zs, xs):
            if z == 1.0 and x == 1.0:
                k += 1
        signs.append(k % 2 != 0)
    return signs


def get_paulis_from_stabilizer_matrix(stabilizer_matrix):
    paulis = []
    nump = len(stabilizer_matrix[0])
    numq = len(stabilizer_matrix) // 2
    for j in range(nump):
        p = ""
        for i in range(numq):
            if stabilizer_matrix[i,j] == stabilizer_matrix[i + numq,j] == 1:
                p += "Y"
            elif stabilizer_matrix[i,j] == 1:
                p += "Z"
            elif stabilizer_matrix[i+numq,j] == 1:
                p += "X"
            else:
                p += "I"
        paulis.append(PauliString(p)._pauli)
    return paulis


def get_measurement_circuit(stabilizer_matrix):
    numq = len(stabilizer_matrix) // 2 # number of qubits
    nump = len(stabilizer_matrix[0]) # number of paulis
    z_matrix = stabilizer_matrix.copy()[:numq]
    x_matrix = stabilizer_matrix.copy()[numq:]

    measurement_circuit = cirq.Circuit()
    qreg = cirq.LineQubit.range(numq)

    # Find a combination of rows to make X matrix have rank nump
    for row_combination in product(['X', 'Z'], repeat=numq):
        candidate_matrix = np.array([
            z_matrix[i] if c=="Z" else x_matrix[i] for i, c in enumerate(row_combination)
        ])

        # Apply Hadamards to swap X and Z rows to transform X matrix to have rank nump
        if np.linalg.matrix_rank(candidate_matrix) == nump:
            for i, c in enumerate(row_combination):
                if c == "Z":
                    z_matrix[i] = x_matrix[i]
                    measurement_circuit.append(cirq.H.on(qreg[i]))
            x_matrix = candidate_matrix
            break
    
    for j in range(nump):
        if x_matrix[j,j] == 0:
            i = j + 1
            while x_matrix[i,j] == 0:
                i += 1

            x_row = x_matrix[i].copy()
            x_matrix[i] = x_matrix[j]
            x_matrix[j] = x_row

            z_row = z_matrix[i].copy()
            z_matrix[i] = z_matrix[j]
            z_matrix[j] = z_row

            measurement_circuit.append(cirq.SWAP.on(qreg[j], qreg[i]))

        for i in range(j + 1, numq):
            if x_matrix[i,j] == 1:
                x_matrix[i] = (x_matrix[i] + x_matrix[j]) % 2
                z_matrix[j] = (z_matrix[j] + z_matrix[i]) % 2

                measurement_circuit.append(cirq.CNOT.on(qreg[j], qreg[i]))

    for j in range(nump-1, 0, -1):
        for i in range(j):
            if x_matrix[i, j] == 1:
                x_matrix[i] = (x_matrix[i] + x_matrix[j]) % 2
                z_matrix[j] = (z_matrix[j] + z_matrix[i]) % 2

                measurement_circuit.append(cirq.CNOT.on(qreg[j], qreg[i]))

    for i in range(nump):
        if z_matrix[i,i] == 1:
            z_matrix[i,i] = 0
            measurement_circuit.append(cirq.S.on(qreg[i]))
        
        for j in range(i):
            if z_matrix[i,j] == 1:
                z_matrix[i,j] = 0
                z_matrix[j,i] = 0
                measurement_circuit.append(cirq.CZ.on(qreg[j], qreg[i]))

    for i in range(nump):
        row = x_matrix[i].copy()
        x_matrix[i] = z_matrix[i]
        z_matrix[i] = row

        measurement_circuit.append(cirq.H.on(qreg[i]))

    return measurement_circuit, np.concatenate((z_matrix, x_matrix))


def get_measurement_circuit_tcc(stabilizer_matrix, distance):
    numq = len(stabilizer_matrix) // 2 # number of qubits
    nump = len(stabilizer_matrix[0]) # number of paulis
    z_matrix = stabilizer_matrix.copy()[:numq]
    x_matrix = stabilizer_matrix.copy()[numq:]

    measurement_circuit = cirq.Circuit()
    qreg = cirq.LineQubit.range(numq)

    # Compute a combination of rows to make X matrix have rank (mod 2) nump
    row_combination_pattern = ""
    pi = 0
    while len(row_combination_pattern) < numq:
        row_combination_pattern = row_combination_pattern + "XXXZXZ" + "XZXZXZ"*pi
        pi += 1
    row_combination = row_combination_pattern[:numq-distance+2] + "Z"*(distance-2)
    candidate_matrix = np.array([
        z_matrix[i] if c=="Z" else x_matrix[i] for i, c in enumerate(row_combination)
    ])

    # Apply Hadamards to swap X and Z rows to transform X matrix to have rank nump
    if np.linalg.matrix_rank(candidate_matrix) == nump:
        for i, c in enumerate(row_combination):
            if c == "Z":
                z_matrix[i] = x_matrix[i]
                measurement_circuit.append(cirq.H.on(qreg[i]))
        x_matrix = candidate_matrix
    
    for j in range(nump):
        if x_matrix[j,j] == 0:
            i = j + 1
            while True:
                if np.isclose(x_matrix[i,j], 0.0):
                    i += 1
                else:
                    break

            x_row = x_matrix[i].copy()
            x_matrix[i] = x_matrix[j]
            x_matrix[j] = x_row

            z_row = z_matrix[i].copy()
            z_matrix[i] = z_matrix[j]
            z_matrix[j] = z_row

            measurement_circuit.append(cirq.SWAP.on(qreg[j], qreg[i]))

        for i in range(j + 1, numq):
            if x_matrix[i,j] == 1:
                x_matrix[i] = (x_matrix[i] + x_matrix[j]) % 2
                z_matrix[j] = (z_matrix[j] + z_matrix[i]) % 2

                measurement_circuit.append(cirq.CNOT.on(qreg[j], qreg[i]))

    for j in range(nump-1, 0, -1):
        for i in range(j):
            if x_matrix[i, j] == 1:
                x_matrix[i] = (x_matrix[i] + x_matrix[j]) % 2
                z_matrix[j] = (z_matrix[j] + z_matrix[i]) % 2

                measurement_circuit.append(cirq.CNOT.on(qreg[j], qreg[i]))

    for i in range(nump):
        if z_matrix[i,i] == 1:
            z_matrix[i,i] = 0
            measurement_circuit.append(cirq.S.on(qreg[i]))
        
        for j in range(i):
            if z_matrix[i,j] == 1:
                z_matrix[i,j] = 0
                z_matrix[j,i] = 0
                measurement_circuit.append(cirq.CZ.on(qreg[j], qreg[i]))

    for i in range(nump):
        row = x_matrix[i].copy()
        x_matrix[i] = z_matrix[i]
        z_matrix[i] = row

        measurement_circuit.append(cirq.H.on(qreg[i]))

    return measurement_circuit, np.concatenate((z_matrix, x_matrix))


def diagonalize_pauli_strings(
    paulis: List[cirq.PauliString], qs: List[cirq.Qid]
) -> Tuple[cirq.Circuit, List[cirq.PauliString]]:
    """Diagonalize a set of Pauli strings, returning the diagonalizing
    circuit and the list of diagonalized strings."""

    stabilizer_matrix = get_stabilizer_matrix_from_paulis(paulis, qs)
    signs = get_stabilizer_matrix_signs(stabilizer_matrix)
    measurment_circuit, _ = get_measurement_circuit(stabilizer_matrix)
    conjugated_strings: List[cirq.PauliString] = []
    for sign, pstring in zip(signs, paulis):
    #    conjugated_string = pstring.after(measurment_circuit)
    #    if sign:
    #        conjugated_strings.append(-1.0 * conjugated_string)
    #    else:
    #        conjugated_strings.append(conjugated_string)
        conjugated_string = pstring.after(measurment_circuit)
        conjugated_strings.append(conjugated_string)
    return measurment_circuit, conjugated_strings
