from pyscf import scf, gto, fci
import openfermion as of

def main():
    molec = "LiH"
    geometry = of.chem.geometry_from_pubchem(molec)
    basis = "sto-3g"
    mol = gto.Mole()
    mol.build(atom=geometry, basis=basis)
    mol_rhf = scf.RHF(mol)
    mol_rhf.run()
    mol_fci = fci.FCI(mol_rhf)
    mol_fci.run()
    print(f"FCI energy: {mol_fci.e_tot}")

if __name__ == "__main__":
    main()
