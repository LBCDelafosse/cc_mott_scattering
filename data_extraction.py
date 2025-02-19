import numpy as np
import scipy.constants as cst
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit


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
    show: bool, optional
        Default: True. If True, displays the graph.
    figname: str, optional
        Default: None. If a string is provided, it will save the figure under the given name.
    """

    nb_curves = len(yaxes)
    plt.rcParams.update({
        "font.family": "serif",   # Type of font
        "xtick.labelsize": "20",  # Size of the xtick label
        "ytick.labelsize": "20",  # Size of the ytick label
        "lines.linewidth": "3",   # Width of the curves
        "legend.framealpha": "1", # Transparency of the legend frame
        "legend.fontsize": "23",  # Size of the legend
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

if __name__ == "__main__":
    data_file = "Exp/Beam2_90Deg_PosA/CardStella_0122_histo_V1.asc"
    det_solid_angle = 2*np.pi * (1. - 1. / np.sqrt(1. + (0.5/123.)**2)) #solid angle occupied by the detector
    target_density = 0.8 * 6.022e23 / 197. #gold target density in SI units
    nb_faraday = 10149
    data = open_data(data_file)
    #plot_data(data)

    fitted_mean, fitted_std_dev, fitted_amplitude, fitted_slope, fitted_intercept, covariance = fit(data, 6593, 10000, 7.2e3, 50., 50., 0., 2.)
    print(fitted_mean, fitted_std_dev, fitted_amplitude, fitted_slope, fitted_intercept, covariance)
    nb_counts = fitted_amplitude * fitted_std_dev * np.sqrt(np.pi) #number of scattered nuclei in the detector
    print(nb_counts)
    y_fit = fit_function(data[:, 0], fitted_mean, fitted_std_dev, fitted_amplitude, fitted_slope, fitted_intercept)
    cross_section = experimental_cross_section(data, nb_counts, nb_faraday, det_solid_angle, target_density, threshold=None, display=False)
    print(cross_section)

    plt.plot(data[:, 0], data[:,1], color='C0')
    plt.plot(data[:, 0], y_fit, 'r-', label='Fitted Curve')

    # Add labels and legend
    plt.xlabel('Bins')
    plt.ylabel('Number of counts')
    plt.legend()

    # Show the plot
    plt.show()
    #plt.savefig('../Fit.pdf')


