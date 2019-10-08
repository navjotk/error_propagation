from util import from_hdf5, plot_field, write_results
import numpy as np
import h5py

from devito import Function, TimeFunction, clear_cache
from examples.seismic import AcquisitionGeometry, Receiver
from examples.checkpointing.checkpoint import CheckpointOperator, DevitoCheckpoint
from simple import overthrust_setup
from skimage.restoration import denoise_tv_chambolle
from scipy.optimize import minimize, Bounds, least_squares
from pyrevolve import Revolver



filename = "overthrust_3D_initial_model_2D.h5"


def load_shot(num):
    basepath = "shots"

    filename = "%s/shot_%d.h5" % (basepath, num)

    with h5py.File(filename) as f:
        data = f['data'][()]
        src_coords = f['src_coords'][()]
    return data, src_coords

def fwi_gradient(vp_in, model, geometry):
    print("FWI/Gradient called")
    # Create symbols to hold the gradient and residual
    grad = Function(name="grad", grid=model.grid)
    vp = Function(name="vp", grid=model.grid)
    residual = Receiver(name='rec', grid=model.grid,
                        time_range=geometry.time_axis, 
                        coordinates=geometry.rec_positions)
    objective = 0.
    vp_in = vec2mat(vp_in)
    global iter
    iter += 1
    plot_field(vp_in, output_file="model%d.png"%iter)
    
    assert(model.vp.shape == vp_in.shape)
    vp.data[:] = vp_in[:]
    # Creat forward wavefield to reuse to avoid memory overload
    solver = overthrust_setup(filename,datakey="m0")
    u0 = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=4,
                      save=geometry.nt)
    for i in range(nshots):
        # Important: We force previous wavefields to be destroyed,
        # so that we may reuse the memory.
        clear_cache()
        
        true_d, source_location = load_shot(i)

        # Update source location
        solver.geometry.src_positions[0, :] = source_location[:]
        
        # Compute smooth data and full forward wavefield u0
        u0.data.fill(0.)
        
        smooth_d, _, _ = solver.forward(vp=vp, save=True, u=u0)
        
        # Compute gradient from data residual and update objective function 
        residual.data[:] = smooth_d.data[:] - true_d[:]
        
        objective += .5*np.linalg.norm(residual.data.flatten())**2
        solver.gradient(rec=residual, u=u0, vp=vp, grad=grad)
    grad.data[:] /= np.max(np.abs(grad.data[:]))
    print("Objective value: %f"%objective)
    return objective, -np.ravel(grad.data).astype(np.float64)



def fwi_gradient_checkpointed(vp_in, model, geometry, n_checkpoints=10):
    print("Checkpointed FWI/Gradient called")
    # Create symbols to hold the gradient and residual
    grad = Function(name="grad", grid=model.grid)
    vp = Function(name="vp", grid=model.grid)
    smooth_d = Receiver(name='rec', grid=model.grid,
                        time_range=geometry.time_axis, 
                        coordinates=geometry.rec_positions)
    residual = Receiver(name='rec', grid=model.grid,
                        time_range=geometry.time_axis, 
                        coordinates=geometry.rec_positions)
    objective = 0.
    time_order = 2
    vp_in = vec2mat(vp_in)
    global iter
    iter += 1
    plot_field(vp_in, output_file="model%d.png"%iter)
    
    assert(model.vp.shape == vp_in.shape)
    vp.data[:] = vp_in[:]
    # Creat forward wavefield to reuse to avoid memory overload
    solver = overthrust_setup(filename,datakey="m0")
    nt = smooth_d.data.shape[0] - 2
    u = TimeFunction(name='u', grid=model.grid, time_order=time_order, space_order=4)
    v = TimeFunction(name='v', grid=model.grid, time_order=time_order, space_order=4)
    wrap_fw = CheckpointOperator(solver.op_fwd(save=False), src=solver.geometry.src, u=u, rec=smooth_d)
    wrap_rev = CheckpointOperator(solver.op_grad(save=False), u=u, v=v, rec=residual, grad=grad)
    cp = DevitoCheckpoint([u])
    compression_params = None
    wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, nt,
                   compression_params=compression_params)
    for i in range(nshots):
        # Important: We force previous wavefields to be destroyed,
        # so that we may reuse the memory.
        clear_cache()
        
        true_d, source_location = load_shot(i)

        # Update source location
        solver.geometry.src_positions[0, :] = source_location[:]
        
        # Compute smooth data and full forward wavefield u0
        u.data.fill(0.)
        
        wrp.apply_forward()
        
        # Compute gradient from data residual and update objective function 
        residual.data[:] = smooth_d.data[:] - true_d[:]
        
        objective += .5*np.linalg.norm(residual.data.flatten())**2
        wrp.apply_reverse()
    grad.data[:] /= np.max(np.abs(grad.data[:]))
    print("Objective value: %f"%objective)
    return objective, -np.ravel(grad.data).astype(np.float64)


tn = 4000
nshots = 20


model = from_hdf5(filename, datakey="m0", dtype=np.float32, space_order=2, nbpml=40)
spacing = model.spacing
shape = model.vp.shape
nrec = shape[0]
src_coordinates = np.empty((1, len(spacing)))
src_coordinates[0, :] = np.array(model.domain_size) * .5
if len(shape) > 1:
    src_coordinates[0, -1] = model.origin[-1] + 2 * spacing[-1]

rec_coordinates = np.empty((nrec, len(spacing)))
rec_coordinates[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
if len(shape) > 1:
    rec_coordinates[:, 1] = np.array(model.domain_size)[1] * .5
    rec_coordinates[:, -1] = model.origin[-1] + 2 * spacing[-1]

# Create solver object to provide relevant operator
geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates,
                                   t0=0.0, tn=tn, src_type='Ricker', f0=0.008)


def apply_box_constraint(vp):
    tv = False
    vp = np.clip(vp, 1.4, 6.1)
    if tv:
        vp = denoise_tv_chambolle(vp, weight=50)
    return vp

iter = 0

def mat2vec(mat):
    print("mat2vec")
    return np.ravel(mat)

def vec2mat(vec):
    print("Vec2Mat", vec.shape)
    if vec.shape == model.vp.shape:
        return vec
    return np.reshape(vec, model.vp.shape)

def f_only(x, model, geometry):
    print("f_only")
    f, g = fwi_gradient(x, model, geometry)
    return f

def g_only(x, model, geometry):
    print("g_only")
    f, g = fwi_gradient(x, model, geometry)
    return g

vmax = np.ones(model.vp.shape) * 6.5
vmin = np.ones(model.vp.shape) * 1.3

vmax[:, 0:20+model.nbpml] = model.vp.data[:, 0:20+model.nbpml]
vmin[:, 0:20+model.nbpml] = model.vp.data[:, 0:20+model.nbpml]
b = Bounds(mat2vec(vmin), mat2vec(vmax))


assert(fwi_gradient_checkpointed(mat2vec(model.vp.data), model, geometry) == fwi_gradient(mat2vec(model.vp.data), model, geometry))


solution_object = minimize(fwi_gradient, mat2vec(model.vp.data), args=(model, geometry), jac=True, method='L-BFGS-B', bounds=b, options={'disp':True})


true_model = from_hdf5("overthrust_3D_true_model_2D.h5", datakey="m", dtype=np.float32, space_order=2, nbpml=40)


error_norm = np.linalg.norm(true_model.vp.data - vec2mat(solution_object.x))
print(error_norm)

data = {'error_norm': error_norm}
write_results(data, "fwi_experiment.csv")
