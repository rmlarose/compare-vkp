import cirq
import openfermion as of
import quimb
import quimb.tensor as qtn
import tensor_network_common as tc

def main():
    hamiltonian = of.jordan_wigner(
            of.get_fermion_operator(
        of.chem.MolecularData(filename="monomer_eqb.hdf5").get_molecular_hamiltonian()
        )
    )
    nqubits = of.utils.count_qubits(hamiltonian)
    nterms  = len(hamiltonian.terms)

    print(f"Hamiltonian acts on {nqubits} qubit(s) and has {nterms} term(s).")

    ham_cirq = of.transforms.qubit_operator_to_pauli_sum(hamiltonian)
    qs = cirq.LineQubit.range(nqubits)
    max_bond = 10_000
    ham_mpo = tc.pauli_sum_to_mpo(ham_cirq, qs, max_bond)

    bond_dims = [ham_mpo.ind_size(b) for b in ham_mpo.inner_inds()]
    bond_dim = max(bond_dims)
    print(f"MPO has maximum bond dimension {bond_dim}")

    quimb.utils.save_to_disk(ham_mpo, "water_mpo.data")

if __name__ == "__main__":
    main()
