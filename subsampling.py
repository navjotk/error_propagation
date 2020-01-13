import numpy as np

from devito import ConditionalDimension, TimeFunction, Operator, Eq, solve, Inc, Function
from examples.seismic import Model, TimeAxis, Receiver, RickerSource

factor = 4  # subsequent calculated factor

t0 = 0.  # Simulation starts a t=0
tn = 2000.  # Simulation lasts tn milliseconds

shape = (100, 100)
origin = (0., 0.)

spacing = (15.,15.)

space_order = 4

vp = np.empty(shape, dtype=np.float32)
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
depth = rec.coordinates.data[:, 1]  # Depth is 20m

time_subsampled = ConditionalDimension(
    't_sub', parent=model.grid.time_dim, factor=factor)
usave = TimeFunction(name='usave', grid=model.grid, time_order=2, space_order=space_order,
                     save=(nt + factor - 1) // factor, time_dim=time_subsampled)
print(time_subsampled)


u = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=space_order)
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


rev_pde = model.m * v.dt2 - v.laplace + model.damp * v.dt
rev_stencil = Eq(v.backward, solve(rev_pde, v.backward))

gradient_update = Inc(grad, - usave.dt2 * v)

s = model.grid.stepping_dim.spacing

receivers = rec.inject(field=v.backward, expr=rec*s**2/model.m)
rev_op = Operator([rev_stencil] + receivers + [gradient_update],
               subs=model.spacing_map)

print(fwd_op)

print(fwd_op.arguments())

fwd_op(time=nt - 2, dt=model.critical_dt)
print("****")

print(rev_op.arguments())

print(rev_op)

rev_op(dt=model.critical_dt)
