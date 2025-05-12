from typing import List, Generator
import h5py
import numpy as np
from scipy.optimize import minimize
import cirq
import openfermion as of
from krylov_common import load_xyz_hamiltonian

def rot_x_layer(qs: List[cirq.Qid], theta: np.ndarray) -> Generator[cirq.Operation, None, None]:
    """Layer of x rotations."""

    assert theta.size == len(qs)
    for i, q in enumerate(qs):
        yield cirq.rx(theta[i]).on(q)


def cz_layer(qs: List[cirq.Qid], even: bool) -> Generator[cirq.Operation, None, None]:
    """A layer of cz gates, acting on paris of adjacent qubits on the line.
    
    Arguments:
    even - if even is True, then CZ will act on qubits 0 and 1, then 2 and 3, etc.
    otherwise, it will act on 1 and 2, 3 and 4, etc.
    
    Returns:
    operations - List of operations for the CZ chain."""

    for i in range(len(qs)):
        if even and i % 2 == 0 and i != len(qs) - 1:
            yield cirq.CZ(qs[i], qs[i+1])
        if not even and i % 2 != 0 and i != len(qs) - 1:
            yield cirq.CZ(qs[i], qs[i+1])


def vqe_circuit(qs: List[cirq.Qid], theta: np.ndarray) -> cirq.Circuit:
    """Build the VQE circuit using a brick wall Ansatz.
    We will have alternating layers of single-qubit x rotations and layers of CZ gates.
    
    Arguments:
    qs - List of qubits to act on.
    theta - a (number of layers) by (number of parameters) array of float parameters.
    
    Returns:
    The Ansatz circuit."""

    assert theta.shape[1] == len(qs)

    ckt = cirq.Circuit()
    for i in range(theta.shape[0]):
        ckt += rot_x_layer(qs, theta[i, :])
        ckt += cz_layer(qs, i % 2 == 0)
    return ckt


def main():
    ham = load_xyz_hamiltonian()
    ham_cirq = of.transforms.qubit_operator_to_pauli_sum(ham)
    nq = of.utils.count_qubits(ham)
    qs = cirq.LineQubit.range(nq)

    nrounds = 2
    theta = np.random.rand(2, len(qs))
    theta_vec = theta.reshape(theta.size)

    def energy_callback(theta_vec: np.ndarray) -> float:
        theta = theta_vec.reshape((nrounds, len(qs)))
        ckt = vqe_circuit(qs, theta)
        sim = cirq.Simulator()
        result = sim.simulate_expectation_values(ckt, [ham_cirq])
        energy = result[0].real
        return energy
    
    opt_result = minimize(energy_callback, theta_vec, method='Nelder-Mead')
    print("Final energy =", opt_result.fun)
    min_ckt = vqe_circuit(qs, opt_result.x.reshape(nrounds, len(qs)))
    min_ckt_qasm = cirq.qasm(min_ckt)
    theta_final = opt_result.x.reshape(nrounds, len(qs))

    # Get the state of the optimized circuit for later reference.
    sim = cirq.Simulator()
    result = sim.simulate(min_ckt)
    state_vector = result.final_state_vector

    f = h5py.File("xyz_vqe.hdf5", "w")
    f.create_dataset("parameters", data=theta_final)
    f.create_dataset("circuit_qasm", data=min_ckt_qasm)
    f.create_dataset("energy", data=opt_result.fun)
    f.create_dataset("state", data=state_vector)

if __name__ == "__main__":
    main()
