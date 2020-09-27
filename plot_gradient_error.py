from argparse import ArgumentParser
import matplotlib
matplotlib.use('Agg')
import tikzplotlib
import matplotlib.pyplot as plt  # noqa
from util import read_csv  # noqa

description = ("Plot the gradient errors")
parser = ArgumentParser(description=description)
parser.add_argument("--filename", type=str, required=True)
parser.add_argument("--replacex", type=str, default=None)
parser.add_argument("--nolog", default=False, action="store_true")
args = parser.parse_args()

filename = args.filename

results = read_csv(filename)

basename = filename.split(".")[0]

xvar = 'ncp'

yvars = ['L0', 'L1', 'L2', 'Linf', 'angle', 'psnr']

replacex = args.replacex

if replacex is not None:
    replacex_filename, replacex_field = replacex.split(":")
    replacex_results = read_csv(replacex_filename)
    x_to_plot = [replacex_results[replacex_field][replacex_results[xvar].index(x)] for x in results[xvar]]  # noqa
    xvar = replacex_field
else:
    x_to_plot = results[xvar]

print(x_to_plot)

for yvar in yvars:
    if not args.nolog:
        plt.xscale('log')
    plt.yscale('log')
    plt.plot(x_to_plot, results[yvar])
    plt.xlabel(xvar)
    plt.ylabel(yvar)
    plt.savefig("%s_%s_%s.pdf" % (basename, yvar, xvar), bbox_inches='tight')
    tikzplotlib.save("%s_%s_%s.tex" % (basename, yvar, xvar))
    plt.clf()
