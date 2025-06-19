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
from openfermion_helper import preprocess_hamiltonian
import tensor_network_common as tnc



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

    # Load fermionic version of the Hamiltonian (no Jordan-Wigner applied)
    hamiltonian = of.utils.load_operator(file_name="owp_631gd_22_ducc.data", data_directory=".")
    # Convert Hamitlonian to an MPO for Block2.
    # Solve for ground state energy with DMRG.
