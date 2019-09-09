import h5py
import numpy as np
import socket
import os.path
import csv


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

def write_results(data, results_file):
    hostname = socket.gethostname()
    if not os.path.isfile(results_file):
        write_header = True
    else:
        write_header = False
    
    data['hostname'] = hostname
    fieldnames = list(data.keys())
    with open(results_file,'a') as fd:
        writer = csv.DictWriter(fd, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(data)

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def error_angle(original, decompressed):
    return angle_between(np.ravel(original), np.ravel(decompressed))


def read_csv(filename):
    results = {}
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for k, v in row.items():
                results_list = results.get(k, [])
                try:
                    v = float(v)
                except ValueError:
                    pass
                results_list.append(v)
                results[k] = results_list
    return results
