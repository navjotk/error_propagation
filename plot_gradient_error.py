import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from util import read_csv

if len(sys.argv) != 2:
    print("Format: plot_gradient_error.py <csv_results_file>")
    sys.exit(0)
    
filename = sys.argv[1]

results = read_csv(filename)

xvar = 'tolerance'

yvars = ['L0', 'L1', 'L2', 'Linf']

for yvar in yvars:
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(results[xvar], results[yvar])
    plt.xlabel(xvar)
    plt.ylabel(yvar)
    plt.savefig("gradient_%s.pdf"%yvar, bbox_inches='tight')
    plt.clf()
