import os
from argparse import ArgumentParser

from devito import Function, TimeFunction
from pyrevolve import Revolver

from examples.seismic import Receiver
from examples.checkpointing.checkpoint import CheckpointOperator, DevitoCheckpoint

from fwi import initial_setup, load_shot
from simple import overthrust_setup
from util import error_L0, error_L1, error_L2, error_Linf, error_angle, write_results


error_metrics = {'L0': error_L0,
                 'L1': error_L1,
                 'L2': error_L2,
                 'Linf': error_Linf,
                 'angle': error_angle}

def calculate_perfect_gradient(i, solver, vp, grad, path_prefix, to=2, so=4):
    true_d, source_location = load_shot(i, path_prefix)

    # Update source location
    solver.geometry.src_positions[0, :] = source_location[:]

    u0 = TimeFunction(name='u', grid=solver.model.grid, time_order=to, space_order=so,
                      save=solver.geometry.nt)

    residual = Receiver(name='rec', grid=solver.model.grid,
                        time_range=solver.geometry.time_axis, 
                        coordinates=solver.geometry.rec_positions)
    smooth_d = Receiver(name='rec', grid=solver.model.grid,
                        time_range=solver.geometry.time_axis, 
                        coordinates=solver.geometry.rec_positions)
        
    smooth_d, _, _ = solver.forward(vp=vp, save=True, u=u0)
        
    # Compute gradient from data residual and update objective function 
    residual.data[:] = smooth_d.data[:] - true_d[:]
    
    solver.gradient(rec=residual, u=u0, vp=vp, grad=grad)
    # The above line does an in-place update so no return required

def calculate_lossy_gradient(i, solver, vp, grad, path_prefix, to=2, so=4, n_checkpoints=1000):
    true_d, source_location = load_shot(i, path_prefix)
    dt = solver.dt
    # Update source location
    solver.geometry.src_positions[0, :] = source_location[:]
        
    # Compute smooth data and full forward wavefield u0
    u = TimeFunction(name='u', grid=solver.model.grid, time_order=to, space_order=so,
                      save=None)
    v = TimeFunction(name='v', grid=solver.model.grid, time_order=to, space_order=so)

    residual = Receiver(name='rec', grid=solver.model.grid,
                        time_range=solver.geometry.time_axis, 
                        coordinates=solver.geometry.rec_positions)
    smooth_d = Receiver(name='rec', grid=solver.model.grid,
                        time_range=solver.geometry.time_axis, 
                        coordinates=solver.geometry.rec_positions)
    fwd_op = solver.op_fwd(save=False)
    rev_op = solver.op_grad(save=False)

    wrap_fw = CheckpointOperator(fwd_op, src=solver.geometry.src, u=u, rec=smooth_d,
                                 vp=vp, dt=dt)
    wrap_rev = CheckpointOperator(rev_op, vp=vp, u=u, v=v, rec=residual, grad=grad, dt=dt)
    cp = DevitoCheckpoint([u])
    nt = smooth_d.data.shape[0] - 2
    wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, nt, compression_params=compression_params)
        
        
    wrp.apply_forward()
        
    # Compute gradient from data residual 
    residual.data[:] = smooth_d.data[:] - true_d[:]
    
    wrp.apply_reverse()
    # The above line does an in-place update so no return required
    
def stacking_experiment(filename, path_prefix, tn=4000, max_shots=40, atol=1e-16,
                        results_file="stacking_experiment_results.csv", to=2, so=4):
    solver = overthrust_setup(path_prefix+"/"+filename, tn=tn, datakey="m0")
    model, geometry, b = initial_setup(path_prefix)
    perfect_grad = Function(name="grad", grid=model.grid)
    lossy_grad = Function(name="grad", grid=model.grid)
    vp = model.vp

    for i in range(max_shots):
        # The following calls are by reference and cumulative
        calculate_perfect_gradient(i, solver, vp, perfect_grad, path_prefix, so=so)
        calculate_lossy_gradient(i, solver, vp, lossy_grad, path_prefix, so=so)

        computed_errors = {}
        for k, v in error_metrics.items():
            computed_errors[k] = v(perfect_grad.data, lossy_grad.data)

        data = computed_errors
        data['shot'] = i
        data['atol'] = atol
        write_results(data, results_file)


if __name__ == "__main__":
    description = ("Experiment to see the effect of stacking on accumulation of errors")
    parser = ArgumentParser(description=description)
    parser.add_argument("-so", "--space_order", default=4,
                        type=int, help="Space order of the simulation")
    parser.add_argument("--ncp", default=None, type=int)
    parser.add_argument("--compression", choices=[None, 'zfp', 'sz', 'blosc'], default=None)
    parser.add_argument("--tolerance", default=6, type=int)
    parser.add_argument("--runmode", choices=["error", "timing"], default="timing")
    parser.add_argument("--nbpml", default=40,
                        type=int, help="Number of PML layers around the domain")
    parser.add_argument("-k", dest="kernel", default='OT2',
                        choices=['OT2', 'OT4'],
                        help="Choice of finite-difference kernel")

    args = parser.parse_args()
    compression_params={'scheme': args.compression, 'tolerance': 10**(-args.tolerance)}

    path_prefix = os.path.dirname(os.path.realpath(__file__))

    stacking_experiment("overthrust_3D_initial_model_2D.h5", path_prefix)
