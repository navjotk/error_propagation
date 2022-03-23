import h5py
from util import plot_field
import numpy as np
import matplotlib.pyplot as plt
import click
import tikzplotlib


def plot_field(data, output_file, threshold=0.01):
    shape = data.shape
    print(shape)
    data[data >= threshold] = threshold
    data[data <= -threshold] = -threshold
    vmax = np.max(data)
    slice_loc = 440
    if len(shape) > 2:
        data = data[slice_loc]

    plt.imshow(np.transpose(data), vmax=vmax, vmin=-vmax, cmap="seismic",
               extent=[0, 20, 0.001*(shape[-1]-1)*25, 0])

    plt.xlabel("X (km)")
    plt.ylabel("Depth (km)")
    cb = plt.colorbar(shrink=.3, pad=.01, aspect=10)
    for i in cb.ax.yaxis.get_ticklabels():
        i.set_fontsize(12)

        cb.set_label('Pressure')

    plt.savefig(output_file, bbox_inches='tight')
    plt.clf()

def plot_histogram(data, output_file):
    data = data.ravel()
    plt.hist(data, log=True)
    plt.savefig(output_file, bbox_inches='tight')
    basename = output_file.split(".")[0]
    tikzplotlib.save("%s.tex" % basename)

@click.command()
@click.option("--input-file", type=(str, str), default=("uncompressed.h5", "data"))
@click.option("--output-file", type=str, default="field.pdf")
@click.option("--plot-type", type=click.Choice(["field", "histogram"]), default="field")
def run(input_file, output_file, plot_type):
    filename, datakey = input_file
    f = h5py.File(filename, 'r')
    data = f[datakey][()]
    if plot_type == "field":
        plot_field(data, output_file=output_file)
    else:
        plot_histogram(data, output_file=output_file)

if __name__ == "__main__":
    run()
