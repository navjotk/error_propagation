import h5py

def to_hdf5(data, filename):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('data', data=data)
