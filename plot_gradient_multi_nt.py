import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
import click


@click.command()
@click.option('--gradient-results-file', help='File containing gradient results')
@click.option('--direct-results-file', default="direct_compression_results.csv", help='File containing direct compression results')
@click.option('--plot-tol/--no-plot-tol', default=True)
@click.option('--plot-cf/--no-plot-cf', default=True)
def draw_plots(gradient_results_file, direct_results_file="direct_compression_results.csv", plot_tol=True, plot_cf=True):
    gradient_results = pd.read_csv(gradient_results_file)

    results_1000 = gradient_results[gradient_results["tn"]==1000][["tolerance", "angle"]].sort_values("tolerance")
    results_2000 = gradient_results[gradient_results["tn"]==2000][["tolerance", "angle"]].sort_values("tolerance")
    results_4000 = gradient_results[gradient_results["tn"]==4000][["tolerance", "angle"]].sort_values("tolerance")

    if plot_tol:
        plt.clf()
        plt.plot(results_1000["tolerance"], results_1000["angle"], "r", label="NT=1000")
        plt.plot(results_2000["tolerance"], results_2000["angle"], "g--", label="NT=2000")
        plt.plot(results_4000["tolerance"], results_4000["angle"], "b:", label="NT=4000")
        plt.xscale('log')
        plt.xlabel("atol")
        plt.ylabel("Angle with perfect gradient (radians)")
        plt.title("Angular deviation of gradient with increasing checkpoint compression")
        plt.legend()
        tikzplotlib.save("gradient_angle_atol.tex")

    if plot_cf:
        direct_results = pd.read_csv(direct_results_file)

        def get_cf_for_tolerance(atol):
            if atol not in range(1, 17):
                atol = - int(np.log10(atol))

            return direct_results.iloc[atol]['cf']

        results_1000["cf"] = [get_cf_for_tolerance(x) for x in results_1000["tolerance"]]
        results_2000["cf"] = [get_cf_for_tolerance(x) for x in results_2000["tolerance"]]
        results_4000["cf"] = [get_cf_for_tolerance(x) for x in results_4000["tolerance"]]
        plt.clf()
        plt.plot(results_1000["cf"], results_1000["angle"], "r", label="NT=1000")
        plt.plot(results_2000["cf"], results_2000["angle"], "g--", label="NT=2000")
        plt.plot(results_4000["cf"], results_4000["angle"], "b:", label="NT=4000")
        plt.xscale('log')
        plt.xlabel("cf")
        plt.ylabel("Angle with perfect gradient (radians)")
        plt.title("Angular deviation of gradient with increasing checkpoint compression")
        plt.legend()
        tikzplotlib.save("gradient_angle_cf.tex")


 if __name__ == '__main__':
     draw_plots()