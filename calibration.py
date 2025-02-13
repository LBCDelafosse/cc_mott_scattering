"""Open, plot and analyze data from CardStella data file."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


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
    Filters the given data by supressing all the data above the given threshold (60 by default).
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

    filter_array = np.where(data[:, 1] > threshold, 1, 0)
    filtered_data[:, 1] = filtered_data[:, 1] * filter_array
    return filtered_data

def get_stats(data: np.array):
    """
    Finds the bin with the most hits and returns the bins index, number of hits and the standard
    deviation.

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

def calibrate(data_file, threshold, display=False):
    """
    Performs the calibration.

    Parameters
    ----------
    data_file: string
        Name of the ASCII file containing the calibration data.
    threshold: int
        The number of hits above which the data is conserved.
    display: boolean, optional
        If true, displays calibration results.

    Returns
    -------
    tuple
        Tuple containing: the function mapping QDC channels to energies, the minimum quadratic deviation to the nominal energy, and the R² coefficient.
    """
    # open the file and plot data
    data = open_data(data_file)

    # filter data and get the peaks
    nominal_bins = []
    quadratic_deviations = [] #mean quadratic deviation from nominal bin
    filtered_cardstella = filter_peaks(data, threshold)
    for idx_start, idx_end in ((12500, 13000), (13500, 14000), (14000, 14500)):
        nominal_bin, nb_counts, std = get_stats(filtered_cardstella[idx_start:idx_end])
        nominal_bins.append([nominal_bin])
        quadratic_deviations.append(std)
        if display:
            print(
                f"bins n°{int(nominal_bin)}\n"
                f".. moyenne nominale : {int(nb_counts)}\n"
                f".. écart-type       : {std:.3f}"
            )

    #plot_data(cardstella, THRESHOLD) #plot the data and show the filter

    # perform the calibration
    nominal_bins = np.array(nominal_bins)
    quadratic_dev = min(quadratic_deviations)
    nominal_energies = np.array([5.16, 5.486, 5.805]) #known energies in MeV
    convert, r2 = linear_regression(nominal_bins, nominal_energies)
    if display: print('slope =', convert(1) - convert(0),'\nintercept =', convert(0))
    quadratic_dev = convert(quadratic_dev) #convert bins into MeV
    if display: print('uncertainty =', quadratic_dev)

    return convert, quadratic_dev, r2
