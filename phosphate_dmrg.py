import argparse
import h5py
import json
import pickle
import h5py
import numpy as np
import cirq
import openfermion as of
from openfermionpyscf import run_pyscf
from openfermion.chem import geometry_from_pubchem, MolecularData
import quimb.tensor as qtn
from openfermion_helper import preprocess_hamiltonian
import tensor_network_common as tnc

def load_hamiltonian(downfold: bool = True, threshold: float = 1e-2) -> of.QubitOperator:
    if not downfold:
        geometry = [
            ("P", (-1.034220, -0.234256,0.672434)),
            ("O", (-1.004065, 0.890081, -0.334695)),
            ("O", (-0.003166, -1.329504, 0.557597)),
            ("O", (-2.065823, -0.232403, 1.765329)),
            ("H", (0.881055, 0.866924, -1.063283)),
            ("O", (1.748944, 0.417505, -1.047631)),
            ("H", (1.477276, -0.378346, -0.549750)),
        ]
        basis = "sto-3g"
        multiplicity = 1
        charge = 1

        molecule = MolecularData(geometry, basis, multiplicity, charge)
        mol = run_pyscf(molecule, run_mp2=True, run_cisd=False, run_ccsd=False, run_fci=False)
        mol.save()
        hamiltonian = MolecularData(filename=molecule.filename)
        hamiltonian = hamiltonian.get_molecular_hamiltonian()
        hamiltonian = of.get_fermion_operator(hamiltonian)
        hamiltonian_openfermion = of.jordan_wigner(hamiltonian)
        hamiltonian_processed = preprocess_hamiltonian(
            hamiltonian_openfermion, drop_term_if=[lambda term: term == ()]
        )  # Drop identity.
    else:
        hamiltonian = of.utils.load_operator(file_name="owp_631gd_22_ducc.data", data_directory=".")
        hamiltonian_processed = of.transforms.jordan_wigner(hamiltonian)
    hamiltonian_processed.compress(abs_tol=threshold)
    return hamiltonian_processed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="JSON input file with paramters.")
    # parser.add_argument("output_file", type=str, help="HDF5 file for simulation output.")
    args = parser.parse_args()

    # Parse argument in JSON file.
    with open(args.input_file, "r", encoding="UTF8") as f:
        input_dict = json.load(f)
    max_bond = input_dict["max_bond"] # Maximum bond dimension of the MPS.
    max_mpo_bond = input_dict["max_mpo_bond"]
    alpha = input_dict["alpha"] # This factor regulates the occupation number.
    n_fermions = input_dict["n_fermions"] # Number of electrons in the system.
    output_fname = input_dict["output_fname"]

    hamiltonian = load_hamiltonian()
    nq = of.utils.count_qubits(hamiltonian)
    print(f"There are {nq} qubits in the Hamiltonian and {len(hamiltonian.terms)} terms.")
    ham_cirq = of.transforms.qubit_operator_to_pauli_sum(hamiltonian)
    qs = cirq.LineQubit.range(nq)
    ham_mpo = tnc.pauli_sum_to_mpo(ham_cirq, qs, max_mpo_bond)
    print("Finished converting to MPO.")

    # Add alpha * (N - N_occ)^2 to the Hamiltonian to ensure the occupation number.
    total_number = tnc.total_number_qubit_operator(nq)
    augmented_hamiltonian = hamiltonian + alpha * (total_number - n_fermions) ** 2
    augmented_hamiltonian_cirq = of.transforms.jordan_wigner(augmented_hamiltonian)
    augmented_hamiltonian_mpo = tnc.pauli_sum_to_mpo(augmented_hamiltonian_cirq, qs, max_mpo_bond)

    dmrg = qtn.DMRG(augmented_hamiltonian_mpo, bond_dims=max_bond)
    converged = dmrg.solve()
    if not converged:
        print("DMRG failed to converge.")
    ground_state = dmrg.state

    energy = tnc.mpo_mps_exepctation(ham_mpo, ground_state)
    print("energy=", energy)

    with open("ground_state.pkl", "wb") as f:
        pickle.dump(ground_state, f)
    
    output_dict = {"input": input_dict, "energy": energy}
    with open(output_fname, "w") as f:
        json.dump(output_dict, f)

if __name__ == "__main__":
    main()
