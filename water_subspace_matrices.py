import h5py
import cirq
import openfermion as of
from kcommute import get_si_sets
from trotter_circuit import trotter_multistep_from_groups
import krylov_common as kc

def main():
    hamiltonian = kc.load_water_hamiltonian()
    nq = of.count_qubits(hamiltonian)
    qs = cirq.LineQubit.range(nq)
    # We start in the same state as the exact diagonalization study.
    state_prep_ckt = kc.hf_ref_circuit(nq, 10)
    # Get the first-order Trotter circuit.
    ham_paulisum = of.transforms.qubit_operator_to_pauli_sum(hamiltonian)
    assert len(ham_paulisum.qubits) == nq
    assert set(ham_paulisum.qubits).issubset(set(qs))
    groups = get_si_sets(ham_paulisum, k=nq)
    tau = 1e-3 # Total time step for evolution unitary.
    steps = 1 # Number of steps per evolution unitary.
    evolution_ckt = trotter_multistep_from_groups(
        groups, qs, tau, steps
    )
    # Convert circuits to QASM.
    prep_ckt_qasm = cirq.qasm(state_prep_ckt, args=cirq.QasmArgs(version="3.0"))
    evolution_ckt_qasm = cirq.qasm(evolution_ckt, args=cirq.QasmArgs(version="3.0"))
    # Compute the subspace matrices.
    d_max = 3
    h, s = kc.subspace_matrices(hamiltonian, state_prep_ckt, evolution_ckt, d_max)
    # Write to file.
    f = h5py.File("subspace_matrices.h5", "w")
    f.create_dataset("tau", data=tau)
    f.create_dataset("steps", data=steps)
    f.create_dataset("prep_circuit_qasm", data=prep_ckt_qasm)
    f.create_dataset("evolution_circuit_qasm", data=evolution_ckt_qasm)
    f.create_dataset("groups", data=str(groups))
    f.create_dataset("d_max", data=d_max)
    f.create_dataset("h", data=h)
    f.create_dataset("s", data=s)
    f.close()

if __name__ == "__main__":
    main()
