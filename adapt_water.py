from typing import Tuple
import numpy as np
import scipy
from scipy.sparse import csc_matrix
import cirq
from qiskit import qpy
import pyscf
import openfermion as of
from openfermion.chem import geometry_from_pubchem, MolecularData
import openfermion_helper
from openfermionpyscf import run_pyscf
from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt
from adaptvqe.pools import FullPauliPool, DVE_CEO
from adaptvqe.matrix_tools import ket_to_vector
from adaptvqe.chemistry import get_hf_det

class QubitHamiltonianWrapper:
    """A wrapper to use openfermion QubitOperators as Hamiltonians for ADAPT-VQE."""

    def __init__(
        self, qubit_operator: of.QubitOperator, num_electrons: int, 
        exact_energy: float, description: str = "custom"
    ) -> None:
        self.operator = qubit_operator
        self.exact_energy = exact_energy
        self.description = description
        # Set the reference state as the Neel state, as in Mafalda's code.
        num_qubits = openfermion_helper.get_num_qubits(qubit_operator)
        ref_det = get_hf_det(num_electrons, num_qubits)
        self.ref_state = csc_matrix(
            ket_to_vector(ref_det), dtype=complex
        ).transpose()
    
    @property
    def ground_energy(self) -> float:
        return self.exact_energy


def load_hamiltonian() -> Tuple[of.QubitOperator, cirq.PauliSum]:
    # Set parameters to make a simple molecule.
    geometry = geometry_from_pubchem("water")
    basis = "sto-3g"
    multiplicity = 1
    charge = 0

    # Make molecule and print out a few interesting facts about it.
    molecule = MolecularData(geometry, basis, multiplicity, charge)

    mol = run_pyscf(molecule, run_mp2=True, run_cisd=True, run_ccsd=True, run_fci=True)
    mol.save()
    water = MolecularData(filename=molecule.filename)
    hamiltonian = water.get_molecular_hamiltonian()
    hamiltonian = of.get_fermion_operator(hamiltonian)
    hamiltonian_openfermion = of.jordan_wigner(hamiltonian)
    hamiltonian = openfermion_helper.preprocess_hamiltonian(
        hamiltonian_openfermion, drop_term_if=[lambda term: term == ()]
    )  # Drop identity.
    return hamiltonian_openfermion, hamiltonian


def get_ground_energy(hamiltonian: of.QubitOperator):
    eval, evec = scipy.sparse.linalg.eigsh(
        of.linalg.get_sparse_operator(hamiltonian),
        k=1,
        which="SA"
    )
    return eval[-1]


def main() -> None:
    hamiltonian_of, hamiltonian_cirq = load_hamiltonian()
    ground_energy = get_ground_energy(hamiltonian_of)
    print(f"Exact energy: {ground_energy}")
    custom_hamiltonian = QubitHamiltonianWrapper(hamiltonian_of, 10, ground_energy)

    pool = DVE_CEO(n=len(hamiltonian_cirq.qubits))
    my_adapt = LinAlgAdapt(
        pool=pool,
        custom_hamiltonian=custom_hamiltonian,
        max_adapt_iter=10,
        recycle_hessian=True,
        tetris=True,
        verbose=True,
        threshold=0.1
    )
    my_adapt.run()
    # Output circuit data.
    data = my_adapt.data
    circuit = pool.get_circuit(data.result.ansatz.indices, data.result.ansatz.coefficients)
    print(f"Final circuit depth: {circuit.depth()}")
    with open("circuit.qpy", "wb") as file:
        qpy.dump(circuit, file)

if __name__ == "__main__":
    main()