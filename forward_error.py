from examples.seismic.acoustic.acoustic_example import acoustic_setup
from math import floor
from pyzfp import compress, decompress
import numpy as np
from devito import TimeFunction, Function
from util import to_hdf5, error_L0, error_L1, error_L2, error_Linf


error_metrics = {'L0': error_L0, 'L1': error_L1, 'L2': error_L2, 'Linf': error_Linf,}

def get_all_errors(original, lossy):
    computed_errors = {}
    for k, v in error_metrics.items():
        computed_errors[k] = v(original, lossy)
    return computed_errors

def get_data(field):
    return field._data

def run_forward_error(space_order=4, kernel='OT4', **kwargs):
    # Setup solver

    solver = acoustic_setup(shape=(10, 10), spacing=(10, 10), nbpml=10, tn=50,
                            space_order=space_order, kernel=kernel, **kwargs)
    
    #solver = overthrust_setup(filename=filename, tn=1000, nbpml=nbpml, space_order=space_order, kernel=kernel, **kwargs)

    # Run for nt/2 timesteps as a warm up
    nt = solver.geometry.time_axis.num
    nt_2 = int(floor(nt/2))

    rec, u, profiler = solver.forward(time=nt_2)

    # Store last timestep

    u_comp = TimeFunction(name='u', grid=solver.model.grid, time_order=2, space_order=solver.space_order)
    u_comp.data
    # Compress-decompress with given tolerance

    atol = 0.001

    compressed_u = compress(get_data(u), tolerance=atol, parallel=True)

    mem = get_data(u_comp)
    mem[:] = decompress(compressed_u, mem.shape, mem.dtype, tolerance=atol)


    for i in range(nt_2):
            # Run for i steps (original last time step and compressed version)
        _, u_original, _ = solver.forward(time_m=nt_2, time_M=nt_2+i, u=u)
        _, u_lossy, _ = solver.forward(time_m=nt_2, time_M=nt_2+i, u=u_comp)

        # Compare and report error metrics

        data = get_all_errors(get_data(u_original), get_data(u_lossy))
        data['ntimesteps'] = i
        print(data)

if __name__ == "__main__":
    run_forward_error()
