import numpy as np
from simple import overthrust_setup
from examples.seismic import plot_shotrecord
from util import from_hdf5, to_hdf5


filename = "overthrust_3D_true_model_2D.h5"
nsrc = 40
model = from_hdf5(filename, datakey="m", dtype=np.float32, space_order=2, nbpml=40)
spacing = model.spacing

basename = "shots"

src_locations = np.linspace(0, model.domain_size[0], nsrc)

for i in range(nsrc):
    src_coords = np.empty((1, 2), dtype=np.float32)
    src_coords[0, 0] = model.origin[0] + src_locations[i]
    src_coords[0, 1] = model.origin[1] + 2*spacing[1]

    solver = overthrust_setup(filename, src_coordinates=src_coords, datakey="m")

    rec, u, _ = solver.forward()

    to_hdf5(rec.data, "%s/shot_%d.h5" % (basename, i), additional={'src_coords': src_coords})
    
