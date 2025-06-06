import argparse
import h5py

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("subspace_file", type=str, help="File with subspace matrix data.")
    parser.add_argument("eigenvalue_file", type=str, help="File with eigenvalue data.")
    parser.add_argument("output_file", type=str, help="New, merged data file.")
    args = parser.parse_args()

    new_f = h5py.File(args.output_file, "w")

    # Transfer data from the subspace matrix file.
    subspace_f = h5py.File(args.subspace_file, "r")
    new_f.create_dataset("l", data=subspace_f["l"][()])
    new_f.create_dataset("tau", data=subspace_f["tau"][()])
    new_f.create_dataset("steps", data=subspace_f["steps"][()])
    new_f.create_dataset("ratio", data=subspace_f["ratio"][()])
    new_f.create_dataset("d_max", data=subspace_f["d_max"][()])
    new_f.create_dataset("h", data=subspace_f["h"][:])
    new_f.create_dataset("s", data=subspace_f["s"][:])
    new_f.create_dataset("ref_state", data=subspace_f["ref_state"][:])
    new_f.create_dataset("reference_energy", data=subspace_f["reference_energy"][()])
    subspace_f.close()

    # Transfer data from the Eigenvalue file.
    egv_f = h5py.File(args.eigenvalue_file, "r")
    new_f.create_dataset("eps", data=egv_f["eps"][()])
    egv_f.copy("eigenvalues", new_f)
    egv_f.close()

    new_f.close()

if __name__ == "__main__":
    main()
