import h5py
import pyzfp
from argparse import ArgumentParser
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa
from util import error_L0, error_L1, error_L2, error_Linf, error_psnr, write_results  # noqa


error_metrics = {'L0': error_L0, 'L1': error_L1, 'L2': error_L2,
                 'Linf': error_Linf, 'psnr': error_psnr}

description = ("Script to calculate error on direct compression of wavefields")
parser = ArgumentParser(description=description)

parser.add_argument("--filename", type=str, required=False,
                    default="uncompressed.h5")
parser.add_argument("--plot", type=str, required=False, default="L0",
                    choices=error_metrics.keys())

args = parser.parse_args()

filename = args.filename
plot = args.plot

f = h5py.File(filename, 'r')

field = f['data'][()].astype(np.float64)

tolerances = [10**x for x in range(0, -17, -1)]

error_to_plot = []

for atol in tolerances:
    print("Compressing at tolerance %s" % str(atol))
    compressed = pyzfp.compress(field, tolerance=atol)
    decompressed = pyzfp.decompress(compressed, shape=field.shape,
                                    dtype=field.dtype, tolerance=atol)

    computed_errors = {}
    for k, v in error_metrics.items():
        computed_errors[k] = v(field, decompressed)

    error_function = error_metrics[plot]
    error_to_plot.append(computed_errors[plot])

    computed_errors['tolerance'] = atol
    write_results(computed_errors, 'direct_compression_results.csv')


plt.xscale('log')
plt.yscale('log')
plt.plot(tolerances, error_to_plot)
plt.xlabel("atol")
plt.ylabel(plot)
plt.savefig("direct_%s.pdf" % plot, bbox_inches='tight')
