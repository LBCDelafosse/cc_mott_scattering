"""Open, plot and analyze data from CardStella data file."""

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as cst
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit


# figure setup
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 14,
        "figure.figsize": (8, 5),
        "figure.constrained_layout.use": True,
    }
)

def open_data(filename: str):
    """
    Opens the given file and extract the data in it.

    Parameters
    ----------
    filename: str
        The name of the file to open.

    Returns
    -------
    np.array
        The data extracted.
    """
    with open(filename, "r", encoding="ascii") as file:
        data = [(index + 1, int(value)) for index, value in enumerate(file.readlines())]
    return np.array(data, dtype=np.float64)

def plot_data(data: np.array, threshold: int = None):
    """
    Plots the data on a graph.

    Parameters
    ----------
    data: np.array
        The data to be plotted.
    threshold: int, optional
        By default: None. If a threshold is provided, it will display the filtered data.
    """
    fig, axes = plt.subplots()

    if threshold is None:
        axes.plot(data[:, 0], data[:, 1])
    else:
        above_data = np.where(data[:, 1] > threshold, data[:, 1], np.nan)
        below_data = np.where(data[:, 1] <= threshold, data[:, 1], np.nan)
        axes.plot(data[:, 0], above_data, color="C0", alpha=1)
        axes.plot(data[:, 0], below_data, color="C0", alpha=0.5)

    fig.suptitle("Number of counts per bin")
    axes.set_xlabel("Bins")
    axes.set_ylabel("Number of counts")

    plt.grid(True)
    plt.show()
    #plt.savefig('Calibration_data.pdf')

def filter_peaks(data: np.array, threshold: int = 15):
    """
    Filters the given data by supressing all the data above the given threshold (15 by default).
    This function works on a deep copy of the given array.

    Parameters
    ----------
    data: np.array
        The data to be filtered.
    threshold: int, optional
        By default: 15. The number of hits above which the data is conserved.

    Returns
    -------
    np.array
        The filtered data.
    """
    filtered_data = data.copy()

    if threshold:
        filter_array = np.where(data[:, 1] > threshold, 1, 0)
        filtered_data[:, 1] = filtered_data[:, 1] * filter_array
    return filtered_data

def get_stats(data: np.array):
    """
    Finds the bin with the most hits and returns the bin index, number of hits and the standard deviation.

    Parameters
    ----------
    data: np.array
        The data to be analyzed.

    Returns
    -------
    tuple
        A tuple that contains the bins, the number of hits and the standard deviation.
    """
    nominal_bin, nb_counts = max(data, key=lambda x: x[1]) #get the max

    variance = 0
    normalization_cst = 0
    for hits in data[:, 1]:
        if hits:
            variance += (nb_counts - hits) ** 2
            normalization_cst += 1
    variance /= normalization_cst

    return nominal_bin, nb_counts, np.sqrt(variance)

def linear_regression(nominal_bins, nominal_energies):
    """
    Finds the coefficients of the linear regression used in the calibration.

    Parameters
    ----------
    nominal_bins: numpy array
        List of nominal bins corresponding to the different peaks found during calibration. Each bin number must be given as a one-element list.
    nominal_energies: numpy array
        List of known energies that must be matched to the nominal bins.

    Returns
    -------
    slope: float
        Slope of the linear regression model.
    intercept: float
        Intercept of the linear regression model.
    r2: float
        R² score for the model.
    """
    # linear regression
    model = LinearRegression()
    model.fit(nominal_bins, nominal_energies)

    pred_energies = model.predict(nominal_bins) #predict using the model
    r2 = r2_score(nominal_energies, pred_energies) #calculate the R² score

    # retrieve the coefficients and intercept
    slope = model.coef_[0]
    intercept = model.intercept_

    return lambda x: (slope * x + intercept), r2

def calibrate(data_file, threshold=None, display=False, plot=False):
    """
    Performs the calibration.

    Parameters
    ----------
    data_file: string
        Name of the ASCII file containing the calibration data.
    threshold: int, optional
        Default: None. The number of hits above which the data is conserved.
    display: boolean, optional
        Default: False. If True, displays calibration results.
    plot: boolean, optional
        Default: False. If True, displays a plot of the data.

    Returns
    -------
    tuple
        Tuple containing: the function mapping QDC channels to energies, the minimum quadratic deviation to the nominal energy, and the R² coefficient.
    """
    data = open_data(data_file) #open the file

    # filter data and get the peaks
    nominal_bins = []
    quadratic_deviations = [] #mean quadratic deviation from nominal bin
    filtered_data = filter_peaks(data, threshold)
    for idx_start, idx_end in ((12500, 13000), (13500, 14000), (14000, 14500)):
        nominal_bin, nb_counts, std = get_stats(filtered_data[idx_start:idx_end])
        nominal_bins.append([nominal_bin])
        quadratic_deviations.append(std)
        if display:
            print(
                f"bins n°{int(nominal_bin)}\n"
                f".. moyenne nominale : {int(nb_counts)}\n"
                f".. écart-type       : {std:.3f}"
            )

    if plot: plot_data(data, threshold) #plot the data and show the filter

    # perform the calibration
    nominal_bins = np.array(nominal_bins)
    quadratic_dev = min(quadratic_deviations)
    nominal_energies = np.array([5.16, 5.486, 5.805]) #known energies in MeV
    convert, r2 = linear_regression(nominal_bins, nominal_energies)
    if display: print('slope =', convert(1) - convert(0),'\nintercept =', convert(0))
    quadratic_dev = convert(quadratic_dev) #convert bins into MeV
    if display: print('uncertainty =', quadratic_dev)

    return convert, quadratic_dev, r2

def experimental_cross_section(data, nb_counts, nb_faraday, det_solid_angle, target_density, threshold=None, display=False, plot=False):
    """
    Computes experimental cross-sections.

    Parameters
    ----------
    data: numpy array
        Calibration data.        
    nb_counts: int
        Number of counts in the detector corresponding to scattered nuclei.
    nb_faraday: int
        Number of counts in the Faraday cups.
    det_solid_angle: float
        Solid angle occupied by the detector in the lab frame, in steradians.
    target_density: float
        Number of scattering centres per squared meter.
    threshold: int, optional
        Default: None. The number of hits above which the data is conserved.
    display: boolean, optional
        Default: False. If True, displays calibration results.
    plot: boolean, optional
        Default: False. If True, displays a plot of the data.

    Returns
    -------
    float
        Experimental differential cross-section in mb/sr.
    """
    if plot: plot_data(data) #plot the data
    nb_particles = nb_faraday * 2e-9 / cst.e #number of particles in the incident beam
    if display: print(nb_particles)
    cross_section = nb_counts / (nb_particles * target_density * det_solid_angle) * 1e31 #experimental differential cross-section in mb/sr.

    return cross_section

def fit_function(x, mean, std_dev, amplitude, slope, intercept):
    return amplitude * np.exp(-0.5 * ((x - mean) / std_dev) ** 2) + slope * x + intercept

def fit(data, data_start, data_end, mean, std_dev, amplitude, slope, intercept):
    initial_guess = [mean, std_dev, amplitude, slope, intercept] #initial guess for the parameters
    params, covariance = curve_fit(fit_function, data[data_start:data_end, 0], data[data_start:data_end, 1], p0=initial_guess) #fit the Gaussian function to the histogram data
    fitted_mean, fitted_std_dev, fitted_amplitude, fitted_slope, fitted_intercept = params

    return fitted_mean, fitted_std_dev, fitted_amplitude, fitted_slope, fitted_intercept, covariance

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

'''
    plt.plot(data[:, 0], data[:,1], color='C0')
    plt.plot(data[:, 0], y_fit, 'r-', label='Fitted Curve')

    # Add labels and legend
    plt.xlabel('Bins')
    plt.ylabel('Number of counts')
    plt.legend()

    # Show the plot
    plt.show()
'''

