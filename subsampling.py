import numpy as np

from devito import (ConditionalDimension, TimeFunction, Operator, Eq, solve,
                    Inc, Function)
from examples.seismic import Model, TimeAxis, Receiver, RickerSource

from util import (error_L0, error_L1, error_L2, error_Linf, error_angle,
                  write_results)


def subsampled_gradient(factor=1, tn=2000.):
    t0 = 0.  # Simulation starts a t=0

    shape = (100, 100)
    origin = (0., 0.)

    spacing = (15., 15.)

    space_order = 4

    vp = np.empty(shape, dtype=np.float64)
    vp[:, :51] = 1.5
    vp[:, 51:] = 2.5

    model = Model(vp=vp, origin=origin, shape=shape, spacing=spacing,
                  space_order=space_order, nbl=10)

    dt = model.critical_dt  # Time step from model grid spacing
    time_range = TimeAxis(start=t0, stop=tn, step=dt)
    nt = time_range.num  # number of time steps

    f0 = 0.010  # Source peak frequency is 10Hz (0.010 kHz)
    src = RickerSource(
        name='src',
        grid=model.grid,
        f0=f0,
        time_range=time_range)

    src.coordinates.data[0, :] = np.array(model.domain_size) * .5
    src.coordinates.data[0, -1] = 20.  # Depth is 20m

    rec = Receiver(
        name='rec',
        grid=model.grid,
        npoint=101,
        time_range=time_range)  # new
    rec.coordinates.data[:, 0] = np.linspace(0, model.domain_size[0], num=101)
    rec.coordinates.data[:, 1] = 20.  # Depth is 20m

    save_elements = (nt + factor - 1) // factor

    print(save_elements)

    time_subsampled = ConditionalDimension(
        't_sub', parent=model.grid.time_dim, factor=factor)
    usave = TimeFunction(name='usave', grid=model.grid, time_order=2,
                         space_order=space_order, save=save_elements,
                         time_dim=time_subsampled)

    u = TimeFunction(name="u", grid=model.grid, time_order=2,
                     space_order=space_order)
    pde = model.m * u.dt2 - u.laplace + model.damp * u.dt
    stencil = Eq(u.forward, solve(pde, u.forward))
    src_term = src.inject(
        field=u.forward,
        expr=src * dt**2 / model.m,
        offset=model.nbl)
    rec_term = rec.interpolate(expr=u, offset=model.nbl)

    fwd_op = Operator([stencil] + src_term + [Eq(usave, u)] + rec_term,
                      subs=model.spacing_map)  # operator with snapshots
    v = TimeFunction(name='v', grid=model.grid, save=None,
                     time_order=2, space_order=space_order)
    grad = Function(name='grad', grid=model.grid)

    rev_pde = model.m * v.dt2 - v.laplace + model.damp * v.dt.T
    rev_stencil = Eq(v.backward, solve(rev_pde, v.backward))
    gradient_update = Inc(grad, - usave.dt2 * v)

    s = model.grid.stepping_dim.spacing

    receivers = rec.inject(field=v.backward, expr=rec*s**2/model.m)
    rev_op = Operator([rev_stencil] + receivers + [gradient_update],
                      subs=model.spacing_map)

    fwd_op(time=nt - 2, dt=model.critical_dt)

    rev_op(dt=model.critical_dt, time=nt-16)

    return grad.data


error_metrics = {'L0': error_L0, 'L1': error_L1, 'L2': error_L2,
                 'Linf': error_Linf, 'angle': error_angle}

print("Starting...")

reference_solution = subsampled_gradient(factor=1)
print(np.linalg.norm(reference_solution))
print("Reference solution acquired")

for f in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
    print("Solving for f=%d"%f)
    solution = subsampled_gradient(factor=f)
    computed_errors = {}
    for k, v in error_metrics.items():
        computed_errors[k] = v(solution, reference_solution)
    computed_errors['f'] = f

    write_results(computed_errors, "subsampling_results.csv")
