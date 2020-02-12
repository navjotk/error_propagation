from argparse import ArgumentParser
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from util import read_csv

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
    #plt.xscale('log')
    #plt.yscale('log')
    plt.plot(x_to_plot, results[yvar])
    plt.xlabel(xvar)
    plt.ylabel(yvar)
    plt.savefig("%s_%s.pdf"%(basename, yvar), bbox_inches='tight')
    plt.clf()
