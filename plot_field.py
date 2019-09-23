import h5py
from util import plot_field
import numpy as np
import matplotlib.pyplot as plt


filename = "overthrust_2D_initial_model.h5"

f = h5py.File(filename, 'r')
datakey='m0'

data_m = f[datakey][()]
print(data_m.shape)
data = data_m
shape = data.shape
vmax = np.max(data)

im = plt.imshow(data, vmax=vmax, vmin=0, cmap="GnBu",
           extent = [0, 20, 0.001*(shape[-1]-1)*25, 0])

plt.xlabel("X (km)")
plt.ylabel("Depth (km)")
cb = plt.colorbar(shrink=.3, pad=.01, aspect=10)
for l in cb.ax.yaxis.get_ticklabels():
    l.set_fontsize(12)

    cb.set_label('Pressure')

#plt.savefig(output_file, bbox_inches='tight')
plt.show()
plt.clf()

plot_field(np.transpose(data_m), output_file='test_field.pdf')
