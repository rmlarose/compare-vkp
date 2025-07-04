from typing import List
import argparse
import pickle
import json
import h5py
import numpy as np
import scipy.linalg as la
import cirq
from cirq.contrib.qasm_import import circuit_from_qasm
import qiskit
import openfermion as of
import qiskit
from qiskit import qpy
from quimb.tensor.tensor_1d import MatrixProductState
from kcommute import get_si_sets
from trotter_circuit import trotter_multistep_from_groups
import krylov_common as kc
from convert import cirq_pauli_sum_to_qiskit_pauli_op
from hamlib_helper import read_openfermion_hdf5

def randomly_perturb_state(ref_state: np.ndarray, ratio: float) -> np.ndarray:
    """Add a random ket |r> to the reference state |phi>, giving the superposition
    N * [(1 - ratio) * |phi> + ratio * |r>], where N is a normalization factor."""

    assert 0.0 <= ratio <= 1.0

    r = np.random.rand(*ref_state.shape)
    new_state = (1.0 - ratio) * ref_state + ratio * r
    return new_state / la.norm(new_state)


def perturb_state_with_cb_state(ref_state: np.ndarray, cb_state: List[bool], ratio: float) -> np.ndarray:
    """Perturb the state with a computational basis state, similar to the above."""

    assert 0.0 <= ratio <= 1.0

    # Build a circuit that prepares the C.B. state.
    ckt = cirq.Circuit()
    qs = cirq.LineQubit.range(len(cb_state))
    for i, b in enumerate(cb_state):
        if b:
            ckt.append(cirq.X(qs[i]))
        else:
            ckt.append(cirq.I(qs[i]))
    # Get the C.B. statevector.
    sim = cirq.Simulator()
    result = sim.simulate(ckt)
    cb_vector = result.final_state_vector

    new_state = (1.0 - ratio) * ref_state + ratio * cb_vector
    return new_state / la.norm(new_state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="JSON input file with parameters.")
    parser.add_argument("output_file", type=str, help="Output filename.")
    args = parser.parse_args()

    with open(args.input_file) as f:
        input_dict = json.load(f)
    hamlib_file = input_dict["hamlib_file"]
    hamlib_key = input_dict["hamlib_key"]
    n_occ = input_dict["n_occ"]
    steps = input_dict["steps"]
    d = input_dict["d"]
    tau = input_dict["tau"]

    #hamiltonian = kc.load_water_hamiltonian()
    hamiltonian = read_openfermion_hdf5(hamlib_file, hamlib_key)
    nq = of.count_qubits(hamiltonian)
    qs = cirq.LineQubit.range(nq)

    # Get the first-order Trotter circuit.
    use_paulihedral = True
    ham_paulisum = of.transforms.qubit_operator_to_pauli_sum(hamiltonian)
    if use_paulihedral:
        ham_qiskit = cirq_pauli_sum_to_qiskit_pauli_op(ham_paulisum)
        dt = tau / float(steps)
        ev_gate = qiskit.circuit.library.PauliEvolutionGate(ham_qiskit, time=dt)
        ev_ckt_qiskit = qiskit.QuantumCircuit(nq)
        for _ in range(steps):
            ev_ckt_qiskit.append(ev_gate, range(nq))
    else:
        assert len(ham_paulisum.qubits) == nq
        assert set(ham_paulisum.qubits).issubset(set(qs))
        tau = tau
        steps = steps
        dt = tau / float(steps)
        groups = get_si_sets(ham_paulisum, k=1)
        with open("hubbard_groups.pkl", "wb") as f:
            pickle.dump(groups, f)
        evolution_ckt = trotter_multistep_from_groups(
            groups, qs, tau, steps
        )
        ev_ckt_qasm = evolution_ckt.to_qasm()
        ev_ckt_qiskit = qiskit.QuantumCircuit.from_qasm_str(ev_ckt_qasm)
    with open("hubbard_trotter_ckt.qpy", "wb") as f:
        qpy.dump(ev_ckt_qiskit, f)

    # Use a computational basis state for the ground state.
    ref_ckt = cirq.Circuit()
    for q in qs:
        if q < n_occ:
            ref_ckt.append(cirq.X(q))
        else:
            ref_ckt.append(cirq.I(q))
    sim = cirq.Simulator()
    ref_state = sim.simulate(ref_ckt).final_state_vector
    qubit_map = {i: q for i, q in enumerate(qs)}
    ref_state_energy = ham_paulisum.expectation_from_state_vector(ref_state, qubit_map)

    # Compute the subspace matrices.
    d_max = d
    h, s = kc.subspace_matrices_from_ref_state(hamiltonian, ref_state, ev_ckt_qiskit, d_max)
    # Write to file.
    f = h5py.File(args.output_file, "w")
    f.create_dataset("hamlib_file", data=hamlib_file)
    f.create_dataset("hamlib_key", data=hamlib_key)
    f.create_dataset("n_occ", data=n_occ)
    f.create_dataset("nq", data=nq)
    f.create_dataset("tau", data=tau)
    f.create_dataset("steps", data=steps)
    f.create_dataset("d_max", data=d_max)
    f.create_dataset("h", data=h)
    f.create_dataset("s", data=s)
    f.create_dataset("ref_state", data=ref_state)
    f.create_dataset("reference_energy", data=ref_state_energy)
    f.close()

if __name__ == "__main__":
    main()
