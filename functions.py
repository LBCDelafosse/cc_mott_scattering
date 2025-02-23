import numpy as np
import scipy.constants as cst
import matplotlib.pyplot as plt


def plot(xaxes, yaxes, linetypes, labels, xlabel, ylabel, log=False, show=True, figname=None):
    """
    Plot a graph.

    Parameters
    ----------
    xaxes: list of numpy arrays
        List of arguments for the plot.
    yaxes: list of numpy arrays
        List of curves to plot.
    linetypes: list of strings
        List of instructions indicating the type of curve to plot.
    labels: list of strings
        List of the labels for each curve to plot.
    xlabel: string
        Label of the x axis.
    ylabel: string
        Label of the y axis.
    log:boolean, optional
        Default: False. If true, uses a logarithmic scale for the y-axis.
    """
    nb_curves = len(yaxes)

    plt.rcParams.update({
        "font.family": "serif",   # Type of font
        "xtick.labelsize": "20",  # Size of the xtick label
        "ytick.labelsize": "20",  # Size of the ytick label
        "lines.linewidth": "3",   # Width of the curves
        "legend.framealpha": "1", # Transparency of the legend frame
        "legend.fontsize": "23",  # Size of the legend
        'axes.labelsize' : '23',  # Size of the axes' labels
        "grid.linestyle":"--",    # Grid formed by dashed lines
        "text.usetex": True       # Using LaTex style for text and equation
    })

    fig, ( fig1 ) = plt.subplots( figsize=(8, 6) )

    for index in range(nb_curves):
        fig1.plot(xaxes[index], yaxes[index], linetypes[index], label=labels[index])

    fig1.set_xlabel(xlabel, fontsize=23)
    fig1.set_ylabel(ylabel, fontsize=23)

    if log: fig1.set_yscale('log')
    fig1.legend(loc='best')

    plt.tight_layout()
    if show: plt.show()
    if figname != None: plt.savefig(figname)

