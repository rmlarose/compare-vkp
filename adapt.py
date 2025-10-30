import openfermion as of
from openfermionpyscf import run_pyscf

from adaptvqe.pools import NoZPauliPool
from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt

natoms_vals = [2, 3, 4, 5, 6]

opcounts = []

for natoms in natoms_vals:
    print("Status: natoms =", natoms)
    bond_distance = 1.0
    geometry = [("H", (0, 0, i * bond_distance)) for i in range(natoms)]
    molecule = of.MolecularData(geometry, basis="sto-3g", multiplicity=1, charge=natoms % 2)

    molecule = run_pyscf(molecule, run_fci=True, run_ccsd=True, run_scf=True, run_mp2=True, verbose=True)

    hamiltonian = of.get_fermion_operator(molecule.get_molecular_hamiltonian())
    hamiltonian = of.jordan_wigner(hamiltonian)

    nqubits = of.utils.count_qubits(hamiltonian)
    nterms = len(hamiltonian.terms)
    print(f"Hamiltonian acts on {nqubits} qubit(s) and has {nterms} Pauli terms.")

    pool = NoZPauliPool(molecule)

    adapt = LinAlgAdapt(
        pool=pool,
        molecule=molecule,
        max_adapt_iter=100,
        recycle_hessian=True,
        tetris=True,
        verbose=True,
        threshold=0.001,
    )
    adapt.run()
    data = adapt.data

    # Create the circuit implementing the final ansatz
    qc = pool.get_circuit(data.result.ansatz.indices, data.result.ansatz.coefficients)
    print("Final ansatz circuit:\n")
    print(qc.draw())
    opcounts.append(qc.count_ops())

    # Access the number of CNOTs and CNOT depth at each iteration
    print("Evolution of ansatz CNOT counts: ", data.acc_cnot_counts(pool))
    print("Evolution of ansatz CNOT depths: ", data.acc_cnot_depths(pool))

print(natoms_vals)
print(opcounts)
