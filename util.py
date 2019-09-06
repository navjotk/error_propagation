import h5py
import numpy as np


def to_hdf5(data, filename):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('data', data=data)

def error_norm(original, decompressed, ord=2):
    error_field = original - decompressed
    return np.linalg.norm(np.ravel(error_field), ord)

def error_L0(original, decompressed):
    return error_norm(original, decompressed, 0)

def error_L1(original, decompressed):
    return error_norm(original, decompressed, 1)

def error_L2(original, decompressed):
    return error_norm(original, decompressed, 2)

def error_Linf(original, decompressed):
    return error_norm(original, decompressed, np.inf)
