import h5py
import pyzfp
import numpy as np
import skimage
import matplotlib.pyplot as plt
from util import to_hdf5, error_L0, error_L1, error_L2, error_Linf


filename = "uncompressed.h5"

plot = "L0"


f = h5py.File(filename, 'r')

field = f['data'][()]

tolerances = [10**x for x in range(0, -16, -1)]

print(tolerances)
error_metrics = {'L0': error_L0, 'L1': error_L1, 'L2': error_L2, 'Linf': error_Linf,
                 'psnr': skimage.measure.compare_psnr}

error_to_plot = []

for atol in tolerances:
    print("Compressing at tolerance %s"%str(atol))
    compressed = pyzfp.compress(field, tolerance=atol)
    decompressed = pyzfp.decompress(compressed, shape=field.shape, dtype=field.dtype, tolerance=atol)

    #computed_errors = {}
    #for k, v in error_metrics.items():
    #    computed_errors[k] = v(field, decompressed)
    #print(computed_errors)
    error_function = error_metrics[plot]
    error_to_plot.append(error_function(field, decompressed))



plt.xscale('log')
plt.yscale('log')
plt.plot(tolerances, error_to_plot)
plt.xlabel("atol")
plt.ylabel(plot)
plt.savefig("%s.pdf"%plot, bbox_inches='tight')

