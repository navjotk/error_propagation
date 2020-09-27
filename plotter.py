import matplotlib.pyplot as plt
import tikzplotlib
from util import read_csv
import click


@click.command()
@click.option('--filename', help='File containing results')
@click.option('--basename', help='Base name for output files')
@click.option('--xvar', help='X variable to plot')
@click.option('--yvar', help='Y variable to plot')
@click.option('--xlog/--no-xlog', default=False)
@click.option('--ylog/--no-ylog', default=False)
@click.option('--hline', help='Coordinate for additional horizontal line, if needed', default=None)
def plot(filename, basename, xvar, yvar, xlog, ylog, hline):
    results = read_csv(filename)

    if xlog:
        plt.xscale('log')
    if ylog:
        plt.yscale('log')
    plt.plot(results[xvar], results[yvar])
    if hline:
        plt.axhline(hline)
    plt.xlabel(xvar)
    plt.ylabel(yvar)
    plt.savefig("%s_%s_%s.pdf" % (basename, yvar, xvar), bbox_inches='tight')
    tikzplotlib.save(("%s_%s_%s.tex" % (basename, yvar, xvar)))
    plt.clf()


if __name__ == '__main__':
    plot()
