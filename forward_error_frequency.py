from argparse import ArgumentParser
from math import floor
from pyzfp import compress, decompress
import numpy as np
import os
from devito import TimeFunction, clear_cache
from util import error_L0, error_L1, error_L2, error_Linf, error_psnr, write_results
from simple import overthrust_setup


error_metrics = {'L0': error_L0, 'L1': error_L1, 'L2': error_L2,
                 'Linf': error_Linf, 'PSNR': error_psnr}


def get_all_errors(original, lossy):
    computed_errors = {}
    for k, v in error_metrics.items():
        computed_errors[k] = v(original, lossy)
    return computed_errors


def get_data(field):
    return field._data.ravel()


def run_forward_error(filename, space_order=4, kernel='OT4', tolerance=1e-6,
                      nbpml=10, dtype=np.float64, **kwargs):
    # Setup solver

    solver = overthrust_setup(filename=filename, tn=2000, nbpml=nbpml,
                              space_order=space_order, kernel=kernel,
                              dtype=dtype, **kwargs)

    
    nt = solver.geometry.time_axis.num
    nt_2 = int(floor(nt/2))

    # Run for nt/2 timesteps as a warm up
    rec, u, profiler = solver.forward(time=nt_2)
    
    # Store last timestep

    u_comp = TimeFunction(name='u', grid=solver.model.grid, time_order=2,
                          space_order=solver.space_order)
    u_comp.data  # Force memory allocation
    # Compress-decompress with given tolerance

    compressed_u = compress(get_data(u), tolerance=tolerance, parallel=True)

    mem = get_data(u_comp)
    mem[:] = decompress(compressed_u, mem.shape, mem.dtype,
                        tolerance=tolerance)

    # Make new symbols so the data in the symbols above is not changed
    u_copy = TimeFunction(name='u', grid=solver.model.grid, time_order=2,
                              space_order=solver.space_order)
    u_copy.data[:] = u.data[:]
    # Uncompressed/Reference version
    _, u_original, _ = solver.forward(time_m=nt_2, time_M=nt, u=u_copy)

    u_l_copy = TimeFunction(name='u', grid=solver.model.grid, time_order=2,
                            space_order=solver.space_order)
    # Lossy version
    u_l_copy.data[:] = u_comp.data[:]
    _, u_lossy, _ = solver.forward(time_m=nt_2, time_M=nt, u=u_l_copy)

if __name__ == "__main__":
    description = ("Example script for a set of acoustic operators.")
    parser = ArgumentParser(description=description)
    parser.add_argument("-so", "--space_order", default=6,
                        type=int, help="Space order of the simulation")
    parser.add_argument("--compression", choices=[None, 'zfp', 'sz', 'blosc'], default='zfp')
    parser.add_argument("--tolerance", default=6, type=int)
    parser.add_argument("--nbpml", default=40,
                        type=int, help="Number of PML layers around the domain")
    parser.add_argument("-k", dest="kernel", default='OT2',
                        choices=['OT2', 'OT4'],
                        help="Choice of finite-difference kernel")
    parser.add_argument("-dse", default="advanced",
                        choices=["noop", "basic", "advanced",
                                 "speculative", "aggressive"],
                        help="Devito symbolic engine (DSE) mode")
    parser.add_argument("-dle", default="advanced",
                        choices=["noop", "advanced", "speculative"],
                        help="Devito loop engine (DLE) mode")
    args = vars(parser.parse_args())
    path_prefix = os.path.dirname(os.path.realpath(__file__))
    args['tolerance'] = 10**(-args['tolerance'])
    args['filename'] = '%s/overthrust_3D_initial_model.h5' % path_prefix
    run_forward_error(**args)
