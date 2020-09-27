from argparse import ArgumentParser
import matplotlib
import tikzplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa
from util import read_csv  # noqa

description = ("Plot the errors from subsampling")
parser = ArgumentParser(description=description)
parser.add_argument("--filename", type=str, required=True)
args = parser.parse_args()

filename = args.filename

results = read_csv(filename)

basename = "subsampling"

xvar = 'f'

yvars = ['L1', 'L2', 'Linf', 'angle']

x_to_plot = results[xvar]

for yvar in yvars:
    # plt.xscale('log')
    plt.yscale('log')
    plt.plot(x_to_plot, results[yvar])
    plt.xlabel('cf')
    plt.ylabel(yvar)
    plt.savefig("%s_%s.pdf" % (basename, yvar), bbox_inches='tight')
    tikzplotlib.save(("%s_%s.tex" % (basename, yvar)))
    plt.clf()
