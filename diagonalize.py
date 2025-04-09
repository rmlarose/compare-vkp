from typing import List, Tuple
from itertools import product
from functools import reduce
import numpy as np
import cirq

from mitiq import PauliString
from qiskit.synthesis import stabilizer


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


def get_linearly_independent_set(stabilizer_matrix: np.ndarray) -> np.ndarray:
    """Use the Gaussian elimination to get the linearly-independent set of vectors.
    
    Returns
    independent_columns - Matrix with the linearly independent columns."""

    # Convert to bool for mod 2 arithmetic.
    bool_sm = stabilizer_matrix.astype(bool)
    reduced_matrix = binary_gaussian_elimination(bool_sm)
    # Find the pivot columns.
    next_pivot = 0 # Row of next pivot
    pivot_columns: List[int] = []
    for j in range(reduced_matrix.shape[1]):
        if next_pivot >= reduced_matrix.shape[0]:
            break
        if reduced_matrix[next_pivot, j]:
            pivot_columns.append(j)
            next_pivot += 1
    independent_columns = stabilizer_matrix[:, pivot_columns]
    assert independent_columns.shape[0] == stabilizer_matrix.shape[0], \
        f"{independent_columns.shape[0]} != {stabilizer_matrix.shape[0]}"
    assert independent_columns.shape[1] <= stabilizer_matrix.shape[1], \
        f"{independent_columns.shape[1]} != {stabilizer_matrix.shape[1]}"
    return independent_columns


def binary_gaussian_elimination(matrix: np.ndarray) -> np.ndarray:
    """Do Gaussian elimination on the matrix to get it into RREF."""

    do_print = False

    next_row = 0 # Row that will contain the next pivot.
    mat = matrix.copy()
    if do_print:
        print("Starting:\n", mat)
    for j in range(mat.shape[1]):
        if do_print:
            print(f"On column {j}.")
        # If a row i >= next_row exists s.t. mat[i, j] == True,
        # Swap rows i and next_row.
        found = False
        for i in range(next_row, mat.shape[0]):
            if mat[i, j]:
                if do_print:
                    print(f"Found True value at row {i}.")
                found = True
                if i != next_row:
                    if do_print:
                        print(f"Swapping {i} <-> {next_row}.")
                    # Swap R_i <-> R_j
                    temp = mat[next_row, :].copy()
                    mat[next_row, :] = mat[i, :]
                    mat[i, :] = temp
                    if do_print:
                        print(mat)
                break
            
        if found:
            if do_print:
                print(f"True at element {next_row}, {j}")
            for i in range(next_row+1, mat.shape[0]):
                if mat[i, j]:
                    if do_print:
                        print(f"XORing row {i} with row{j}.")
                    mat[i, :] ^= mat[next_row, :]
                    if do_print:
                        print(mat)
            next_row += 1
    if do_print:
        print("Final reduced matrix\n", mat)
    return mat


def assert_no_zero_column_in_matrix(matrix):
    """Assert that no column in the matrix is all zeros."""

    for j in range(matrix.shape[1]):
        assert not np.all(matrix[:, j] == 0.0), \
            f"Column {j} is all zeros."


def get_measurement_circuit(stabilizer_matrix):
    numq = len(stabilizer_matrix) // 2 # number of qubits
    nump = len(stabilizer_matrix[0]) # number of paulis
    z_matrix = stabilizer_matrix.copy()[:numq]
    x_matrix = stabilizer_matrix.copy()[numq:]

    assert_no_zero_column_in_matrix(stabilizer_matrix)

    if nump > 2 * numq:
        raise ValueError(f"nump = {nump} > 2 * numq = 2 * {numq}. Set might be linearly-dependent.")

    if nump > numq:
        raise ValueError(f"nump = {nump} > numq = {numq}. More columns than qubits.")
    
    if nump < numq:
        raise ValueError("nump = {nump} < numq = {numq}.")

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
    
    for j in range(min(nump, numq)):
        if x_matrix[j,j] == 0:
            # Find i > j s.t. x_matrix[i, j] = 1.0.
            found = False
            for i in range(j + 1, numq):
                if x_matrix[i, j] != 0:
                    found = True
                    break
            #i = j + 1
            #while x_matrix[i,j] == 0:
            #    i += 1

            # If i was found, apply a SWAP i <-> j.
            if found:
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


def is_pauli_diagonal(pstring: cirq.PauliString) -> bool:
    """Tests if a given PauliString is diagonal."""

    for _, pauli in pstring.items():
        if not (pauli == cirq.I or pauli == cirq.Z):
            return False
    return True


def diagonalize_pauli_strings(
    paulis: List[cirq.PauliString], qs: List[cirq.Qid]
) -> Tuple[cirq.Circuit, List[cirq.PauliString]]:
    """Diagonalize a set of Pauli strings, returning the diagonalizing
    circuit and the list of diagonalized strings."""

    stabilizer_matrix = get_stabilizer_matrix_from_paulis(paulis, qs)
    if stabilizer_matrix.shape[1] > stabilizer_matrix.shape[0] // 2:
        print("Using Gram-Schmidt.")
        print(stabilizer_matrix)
        print(f"Matrix has {stabilizer_matrix.shape[1]} columns.")
        reduced_stabilizer_matrix = get_linearly_independent_set(stabilizer_matrix)
        print(f"After, matrix has {reduced_stabilizer_matrix.shape[1]} columns.")
        print(reduced_stabilizer_matrix)
    else:
        reduced_stabilizer_matrix = stabilizer_matrix.copy()
        print("Not using Gram-Schmidt.")
    measurment_circuit, _ = get_measurement_circuit(reduced_stabilizer_matrix)
    conjugated_strings: List[cirq.PauliString] = []
    for pstring in paulis:
        conjugated_string = pstring.after(measurment_circuit)
        conjugated_strings.append(conjugated_string)
        assert is_pauli_diagonal(conjugated_string), \
            f"Pauli string {conjugated_string} is not diagonal. Originally was {pstring}"
    return measurment_circuit, conjugated_strings
