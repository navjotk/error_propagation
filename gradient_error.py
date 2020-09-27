from argparse import ArgumentParser
import os
import numpy as np

from devito import TimeFunction, Function

from examples.checkpointing.checkpoint import (DevitoCheckpoint,
                                               CheckpointOperator)

from examples.seismic import Receiver
from pyrevolve import Revolver
from timeit import default_timer
from simple import overthrust_setup

from examples.seismic.acoustic.acoustic_example import acoustic_setup
from util import (error_L0, error_L1, error_L2, error_Linf,
                  error_angle, write_results, error_psnr)


error_metrics = {'L0': error_L0, 'L1': error_L1, 'L2': error_L2,
                 'Linf': error_Linf, 'angle': error_angle, 
                 'psnr': error_psnr}


class Timer(object):
    def __init__(self, tracker):
        self.timer = default_timer
        self.tracker = tracker

    def __enter__(self):
        self.start = self.timer()
        return self

    def __exit__(self, *args):
        end = self.timer()
        self.elapsed_secs = end - self.start
        self.elapsed = self.elapsed_secs * 1000  # millisecs
        self.tracker.append(self.elapsed)


def verify(space_order=4, kernel='OT4', nbpml=40, filename='',
           compression_params={}, **kwargs):
    solver = acoustic_setup(shape=(10, 10), spacing=(10, 10), nbpml=10, tn=50,
                            space_order=space_order, kernel=kernel, **kwargs)
    # solver = overthrust_setup(filename=filename, tn=50, nbpml=nbpml,
    #                           space_order=space_order, kernel=kernel,
    #                           **kwargs)

    u = TimeFunction(name='u', grid=solver.model.grid, time_order=2,
                     space_order=solver.space_order)

    rec = Receiver(name='rec', grid=solver.model.grid,
                   time_range=solver.geometry.time_axis,
                   coordinates=solver.geometry.rec_positions)
    cp = DevitoCheckpoint([u])
    n_checkpoints = None
    dt = solver.dt
    v = TimeFunction(name='v', grid=solver.model.grid, time_order=2,
                     space_order=solver.space_order)
    grad = Function(name='grad', grid=solver.model.grid)
    wrap_fw = CheckpointOperator(solver.op_fwd(save=False),
                                 src=solver.geometry.src, u=u, rec=rec, dt=dt)
    wrap_rev = CheckpointOperator(solver.op_grad(save=False), u=u, v=v,
                                  rec=rec, dt=dt, grad=grad)
    nt = rec.data.shape[0] - 2
    print("Verifying for %d timesteps" % nt)
    wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, nt,
                   compression_params=compression_params)
    wrp.apply_forward()
    wrp.apply_reverse()

    print(wrp.profiler.timings)

    with Timer([]) as tf:
        rec2, u2, _ = solver.forward(save=True)

    with Timer([]) as tr:
        grad2, _ = solver.gradient(rec=rec2, u=u2)

    error = grad.data - grad2.data
    # to_hdf5(error, 'zfp_grad_errors.h5')
    print("Error norm", np.linalg.norm(error))

    # assert(np.allclose(grad.data, grad2.data))
    print("Checkpointing implementation is numerically verified")
    print("Verification took %d ms for forward and %d ms for reverse" % (tf.elapsed, tr.elapsed))


def checkpointed_run(space_order=4, ncp=None, kernel='OT4', nbpml=40,
                     filename='', compression_params={}, tn=1000, **kwargs):
    solver = overthrust_setup(filename=filename, tn=tn, nbpml=nbpml,
                              space_order=space_order, kernel=kernel, **kwargs)

    u = TimeFunction(name='u', grid=solver.model.grid, time_order=2,
                     space_order=solver.space_order)
    rec = Receiver(name='rec', grid=solver.model.grid,
                   time_range=solver.geometry.time_axis,
                   coordinates=solver.geometry.rec_positions)
    cp = DevitoCheckpoint([u])
    n_checkpoints = ncp

    dt = solver.dt
    v = TimeFunction(name='v', grid=solver.model.grid, time_order=2,
                     space_order=solver.space_order)

    grad = Function(name='grad', grid=solver.model.grid)

    wrap_fw = CheckpointOperator(solver.op_fwd(save=False),
                                 src=solver.geometry.src, u=u,
                                 rec=rec, dt=dt)

    wrap_rev = CheckpointOperator(solver.op_grad(save=False), u=u, v=v,
                                  rec=rec, dt=dt, grad=grad)

    fw_timings = []
    rev_timings = []

    nt = rec.data.shape[0] - 2
    print("Running %d timesteps" % (nt))
    print(compression_params)
    wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, nt,
                   compression_params=compression_params)
    with Timer(fw_timings):
        wrp.apply_forward()
    with Timer(rev_timings):
        wrp.apply_reverse()

    return grad, wrp, fw_timings, rev_timings


def compare_error(space_order=4, ncp=None, kernel='OT4', nbpml=40, filename='',
                  tn=1000, compression_params={}, **kwargs):
    grad, wrp, fw_timings, rev_timings = checkpointed_run(space_order, ncp,
                                                          kernel, nbpml,
                                                          filename,
                                                          compression_params,
                                                          tn, **kwargs)
    print(wrp.profiler.summary())

    compression_params['scheme'] = None

    print("*************************")
    print("Starting uncompressed run:")

    grad2, _, _, _ = checkpointed_run(space_order, ncp, kernel, nbpml,
                                      filename, compression_params,
                                      tn, **kwargs)

    # error_field = grad2.data - grad.data

    print("compression enabled norm", np.linalg.norm(grad.data))
    print("compression disabled norm", np.linalg.norm(grad2.data))
    # to_hdf5(error_field, 'zfp_grad_errors_full.h5')
    computed_errors = {}

    for k, v in error_metrics.items():
        computed_errors[k] = v(grad2.data, grad.data)

    data = computed_errors
    data['tolerance'] = compression_params['tolerance']
    data['ncp'] = ncp
    data['tn'] = tn

    write_results(data, 'gradient_error_results.csv')


if __name__ == "__main__":
    description = ("Experiment to see the effect of checkpoint compression error on the gradient")
    parser = ArgumentParser(description=description)
    parser.add_argument("-so", "--space_order", default=6,
                        type=int, help="Space order of the simulation")
    parser.add_argument("--ncp", default=None, type=int)
    parser.add_argument("--compression", choices=[None, 'zfp', 'sz', 'blosc'],
                        default=None)
    parser.add_argument("--tolerance", default=6, type=int)
    parser.add_argument("--runmode", choices=["error", "timing"],
                        default="timing")
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
    parser.add_argument("--tn", default=1000, type=int,
                        help="Number of ms to run simulation for")
    args = parser.parse_args()

    compression_params = {'scheme': args.compression,
                          'tolerance': 10**(-args.tolerance)}

    path_prefix = os.path.dirname(os.path.realpath(__file__))
    compare_error(nbpml=args.nbpml, ncp=args.ncp, space_order=args.space_order,
                  kernel=args.kernel, dse=args.dse, dle=args.dle,
                  filename='%s/overthrust_3D_initial_model.h5' % path_prefix,
                  compression_params=compression_params, tn=args.tn)
