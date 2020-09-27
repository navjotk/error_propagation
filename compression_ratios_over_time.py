from argparse import ArgumentParser
from contexttimer import Timer

import numpy as np
import csv

from simple import overthrust_setup
from pyzfp import compress, decompress
from util import write_results


def run(tn=4000, space_order=4, kernel='OT4', nbpml=40, tolerance=1e-4, filename='', **kwargs):
    if kernel in ['OT2', 'OT4']:
        solver = overthrust_setup(filename=filename, tn=tn, nbpml=nbpml, space_order=space_order, kernel=kernel, **kwargs)
    else:
        raise ValueError()

    total_timesteps = solver.geometry.src.time_range.num
    u = None
    rec = None
    for t in range(1, total_timesteps-1):
        rec, u, _ = solver.forward(u=u, rec=rec, time_m=t, time_M=t, save=False)
        uncompressed = u._data[t]
        with Timer(factor=1000) as time1:
            compressed = compress(uncompressed, tolerance=tolerance, parallel=True)
        result = {'timestep': t, 'cf': len(uncompressed.tostring())/float(len(compressed)), 'time': time1.elapsed}
        write_results(result, "cf_vs_nt.csv")

    _, u2, _ = solver.forward(save=False)
    assert(u2.shape == u.shape)
    assert(np.all(np.isclose(u2.data, u.data)))


if __name__ == "__main__":
    description = ("Example script for a set of acoustic operators.")
    parser = ArgumentParser(description=description)
    parser.add_argument("-so", "--space_order", default=6,
                        type=int, help="Space order of the simulation")
    parser.add_argument("--nbpml", default=40,
                        type=int, help="Number of PML layers around the domain")
    parser.add_argument("-k", dest="kernel", default='OT2',
                        choices=['OT2', 'OT4', 'TTI'],
                        help="Choice of finite-difference kernel")
    parser.add_argument("-dse", default="advanced",
                        choices=["noop", "basic", "advanced",
                                 "speculative", "aggressive"],
                        help="Devito symbolic engine (DSE) mode")
    parser.add_argument("-dle", default="advanced",
                        choices=["noop", "advanced", "speculative"],
                        help="Devito loop engine (DLE) mode")
    parser.add_argument("-n", default=4000, type=int,
                        help="Simulation Time (ms)")
    args = parser.parse_args()

    run(nbpml=args.nbpml, tn=args.n,
        space_order=args.space_order, kernel=args.kernel,
        dse=args.dse, dle=args.dle, filename='overthrust_3D_initial_model.h5')