from typing import Dict
import argparse
import os
import numpy as np
import h5py
from krylov_common import fill_subspace_matrices

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", type=str, help="Directory containing output files. They must have 'output' in the filename.")
args = parser.parse_args()

data_dir = args.data_dir
output_files = [f for f in os.listdir(data_dir) if 'output' in f]

idx_to_overlaps: Dict[int, complex] = {}
idx_to_mat_elems: Dict[int, complex] = {}
for of in output_files:
    f = h5py.File(data_dir + "/" + of, "r")
    ds = f["d_vals"][:]
    overlaps = f["overlaps"][:]
    mat_elems = f["mat_elems"][:]
    for d, overlap, mat_elem in zip(ds, overlaps, mat_elems):
        idx_to_overlaps[d] = overlap
        idx_to_mat_elems[d] = mat_elem
    f.close()

max_idx = max(idx_to_overlaps.keys())
if set(idx_to_overlaps.keys()) != set(range(max_idx + 1)):
    raise ValueError(f"Keys should be range({max_idx + 1}) but got {sorted(list(idx_to_overlaps.keys()))}.")

overlap_list = [idx_to_overlaps[i] for i in range(max_idx)]
mat_elem_list = [idx_to_mat_elems[i] for i in range(max_idx)]
h, s = fill_subspace_matrices(mat_elem_list, overlap_list)

f_out = h5py.File(data_dir + "/water_subspace.hdf5", "w")
f_out.create_dataset("h", data=h)
f_out.create_dataset("s", data=s)
f_out.close()