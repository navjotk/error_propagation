from examples.seismic.acoustic.acoustic_example import acoustic_setup
from math import floor
from pyzfp import compress, decompress
import numpy as np
from devito import TimeFunction, Function


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

    # Compress-decompress with given tolerance

    print(u.data.flags)

    compressed_u = compress(u.data, tolerance=0.1, parallel=True)

    u_comp.data[:] = decompress(compressed_u, u_comp.shape, u_comp.dtype, tolerance=0.1)

    print(np.linalg.norm(u_comp.data - u.data))


    # for i in range(nt/2)
    # Run for i steps (original last time step and compressed version)
    # Compare and report error metrics

if __name__ == "__main__":
    run_forward_error()
