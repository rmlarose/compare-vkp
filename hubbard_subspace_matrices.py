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
    parser.add_argument("exact_input_file", type=str, help="HDF5 file with ground state.")
    parser.add_argument("output_file", type=str, help="Output filename.")
    args = parser.parse_args()

    with open(args.input_file) as f:
        input_dict = json.load(f)
    l = input_dict["l"] # Number of lattice sites on each side.
    n_occ = input_dict["n_occ"] # Number of occupied orbitals.
    t = input_dict["t"] # Hopping rate
    u = input_dict["u"] # Interaction strength.
    steps = input_dict["steps"]
    ratio = input_dict["ratio"]
    d = input_dict["d"]
    tau = input_dict["tau"]
    max_circuit_bond = input_dict["max_circuit_bond"]
    max_mpo_bond = input_dict["max_mpo_bond"]

    #hamiltonian = kc.load_water_hamiltonian()
    ham_fermi = of.hamiltonians.fermi_hubbard(l, l, t, u, spinless=True)
    hamiltonian = of.transforms.jordan_wigner(ham_fermi)
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

    # For debug purposes: Get the exact ground state from the hubbard_exact.h5 file,
    # and perturb it with a computational basis state.
    f_in = h5py.File(args.exact_input_file, "r")
    ground_state = f_in["ground_state"][:]
    n_occ = round(f_in["number"][()].real)
    f_in.close()
    print(f"{n_occ} occupied states in reference.")
    b = [True] * n_occ + [False] * (nq - n_occ)
    assert 0.0 <= ratio <= 1.0
    ref_state = perturb_state_with_cb_state(ground_state, b, ratio)
    qubit_map = dict(zip(qs, range(len(qs))))
    ref_state_energy = ham_paulisum.expectation_from_state_vector(ref_state, qubit_map)
    print(f"Reference energy = {ref_state_energy}")
    del ground_state

    # Compute the subspace matrices.
    d_max = d
    use_tebd = True
    if not use_tebd:
        h, s = kc.subspace_matrices_from_ref_state(hamiltonian, ref_state, ev_ckt_qiskit, d_max)
    else:
        reference_mps = MatrixProductState.from_dense(ref_state)
        h, s = kc.tebd_subspace_matrices(ham_paulisum, ev_ckt_qiskit, reference_mps,
                                         d_max, max_circuit_bond, max_mpo_bond)
    # Write to file.
    f = h5py.File(args.output_file, "w")
    f.create_dataset("l", data=l)
    f.create_dataset("tau", data=tau)
    f.create_dataset("steps", data=steps)
    f.create_dataset("ratio", data=ratio)
    # f.create_dataset("prep_circuit_qasm", data=prep_ckt_qasm)
    # f.create_dataset("evolution_circuit_qasm", data=evolution_ckt_qasm)
    # f.create_dataset("groups", data=str(groups))
    f.create_dataset("d_max", data=d_max)
    f.create_dataset("h", data=h)
    f.create_dataset("s", data=s)
    f.create_dataset("ref_state", data=ref_state)
    f.create_dataset("reference_energy", data=ref_state_energy)
    f.close()

if __name__ == "__main__":
    main()
