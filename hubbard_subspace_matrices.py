from typing import List
import argparse
import pickle
import h5py
import numpy as np
import scipy.linalg as la
import cirq
import qiskit
import openfermion as of
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
    parser.add_argument("tau", type=float, help="Evolution time for U.")
    parser.add_argument("steps", type=int, help="Number of steps for U.")
    parser.add_argument("d", type=int, help="Subspace dimension")
    parser.add_argument("output_file", type=str, help="Output filename.")
    args = parser.parse_args()

    #hamiltonian = kc.load_water_hamiltonian()
    hamiltonian: of.QubitOperator = kc.load_hubbard_hamiltonian()
    nq = of.count_qubits(hamiltonian)
    qs = cirq.LineQubit.range(nq)

    # We start in the same state as the exact diagonalization study.
    #state_prep_ckt = kc.hf_ref_circuit(nq, 10)
    state_prep_ckt = kc.neel_state_circuit(nq)
    #state_prep_ckt = kc.neel_state_circuit_qiskit(nq)

    # Get the first-order Trotter circuit.
    ham_paulisum = of.transforms.qubit_operator_to_pauli_sum(hamiltonian)
    assert len(ham_paulisum.qubits) == nq
    assert set(ham_paulisum.qubits).issubset(set(qs))
    tau = args.tau
    steps = args.steps
    dt = tau / float(steps)
    # BEGIN DEBUG CODE trying qiskit+PauliHedral
    # ham_qiskit = cirq_pauli_sum_to_qiskit_pauli_op(ham_paulisum)
    # single_step_ckt = qiskit.circuit.library.PauliEvolutionGate(ham_qiskit, time=dt)
    # evolution_ckt = qiskit.QuantumCircuit(nq)
    # for _ in range(steps):
    #     evolution_ckt.compose(single_step_ckt, inplace=True)
    # END DEBUG CODE
    # BEGIN OLD CODE
    groups = get_si_sets(ham_paulisum, k=1)
    with open("hubbard_groups.pkl", "wb") as f:
        pickle.dump(groups, f)
    evolution_ckt = trotter_multistep_from_groups(
        groups, qs, tau, steps
    )
    ev_ckt_qasm = evolution_ckt.to_qasm()
    ev_ckt_qiskit = qiskit.QuantumCircuit.from_qasm_str(ev_ckt_qasm)
    # convert to qisit.
    # BEGIN OLD CODE
    # Convert circuits to QASM.
    #prep_ckt_qasm = cirq.qasm(state_prep_ckt)
    #evolution_ckt_qasm = cirq.qasm(evolution_ckt)

    # For debug purposes: Get the exact ground state from the hubbard_exact.h5 file,
    # and perturb it with a computational basis state.
    f_in = h5py.File("hubbard_exact.h5", "r")
    ground_state = f_in["eigenvectors"][:, 0]
    n_occ = 3
    b = [True] * n_occ + [False] * (nq - n_occ)
    ref_state = perturb_state_with_cb_state(ground_state, b, 1e-1)
    print("distance =", la.norm(ground_state - ref_state))

    # Compute the subspace matrices.
    d_max = args.d
    h, s = kc.subspace_matrices_from_ref_state(hamiltonian, ref_state, ev_ckt_qiskit, d_max)
    # Write to file.
    f = h5py.File(args.output_file, "w")
    f.create_dataset("tau", data=tau)
    f.create_dataset("steps", data=steps)
    # f.create_dataset("prep_circuit_qasm", data=prep_ckt_qasm)
    # f.create_dataset("evolution_circuit_qasm", data=evolution_ckt_qasm)
    # f.create_dataset("groups", data=str(groups))
    f.create_dataset("d_max", data=d_max)
    f.create_dataset("h", data=h)
    f.create_dataset("s", data=s)
    f.close()

if __name__ == "__main__":
    main()
