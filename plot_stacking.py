import pandas as pd
import matplotlib.pyplot as plt
import tikzplotlib
import click


@click.command()
@click.option('--filename', help='File containing the results to plot')
def draw_plots(filename):

    stacking_results = pd.read_csv(filename)

    results_1 = stacking_results[stacking_results["atol"]==1e-1][["shot", "angle"]].sort_values("shot")
    results_4 = stacking_results[stacking_results["atol"]==1e-4][["shot", "angle"]].sort_values("shot")
    results_8 = stacking_results[stacking_results["atol"]==1e-8][["shot", "angle"]].sort_values("shot")
    results_16 = stacking_results[stacking_results["atol"]==1e-16][["shot", "angle"]].sort_values("shot")

    plt.plot(results_1["shot"], results_1["angle"], "-", label="atol=1e-1")
    plt.plot(results_4["shot"], results_4["angle"], "--", label="atol=1e-4")
    plt.plot(results_8["shot"], results_8["angle"], ":", label="atol=1e-8")
    plt.plot(results_16["shot"], results_16["angle"], ".-", label="atol=1e-16")
    plt.xlabel("shots")
    plt.ylabel("Angle with perfect gradient (radians)")
    plt.title("Angular deviation of gradient as a function of number of shots stacked")
    plt.legend()
    plt.savefig("stacking_experiment.pdf", bbox_inches='tight')
    tikzplotlib.save("stacking_experiment.tex")


if __name__ == '__main__':
    draw_plots()
