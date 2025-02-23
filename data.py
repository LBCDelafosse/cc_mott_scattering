import numpy as np
import scipy.constants as cst
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from collision import *


def create_data_from_filename(filename: str):
    """
    Create a Data object by reading an external file.

    Parameters
    ----------
    filename: str
        Name of the ascii file containing the data.

    Returns
    -------
    Data
        The Data instance.
    """
    # open and read the file
    with open(filename, "r", encoding="ascii") as file:
        data = [(index + 1, int(value)) for index, value in enumerate(file.readlines())]
    data = np.array(data)
    return Data(data)

class Data:
    """
    Contain and handle experimental data.
    """

    # Constructors

    def __init__(self, data: np.array, threshold: int = None):
        """
        Default constructor of the Data class.

        Parameters
        ----------
        data: np.array
            Data composed of two columns (bins and numbers of counts).
        threshold:
            Default: None. Else, threshold under which data are not considered.
        """
        self.data = data
        self.threshold = threshold

    # Accessors

    def __getitem__(self, indices):
        """
        Acess the self.data instance attribute.
        """
        return self.data[indices]

    # Modifiers

    def set_threshold(self, threshold: int):
        """
        Set a threshold under which data are not considered.

        Parameters
        ----------
        threshold: int
        """
        self.threshold = threshold

    # Other methods

    def plot_data(self, show: bool = True, figname: str = None):
        """
        Plot the data on a graph.

        Parameters
        ----------
        show: bool, optional
            Default: True. If True, displays the graph.
        figname: str, optional
            Default: None. If a string is provided, it will save the figure under the given name.
        """
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.size": 14,
                "figure.figsize": (8, 5),
                "figure.constrained_layout.use": True,
            }
        )

        fig, axes = plt.subplots()

        if self.threshold is None:
            axes.plot(self.data[:, 0], self.data[:, 1])
        else:
            above_data = np.where(self.data[:, 1] > self.threshold, self.data[:, 1], np.nan)
            below_data = np.where(self.data[:, 1] <= self.threshold, self.data[:, 1], np.nan)
            axes.plot(data[:, 0], above_data, color="C0", alpha=1)
            axes.plot(data[:, 0], below_data, color="C0", alpha=0.5)

        fig.suptitle("Number of counts per bin")
        axes.set_xlabel("Bins")
        axes.set_ylabel("Number of counts")

        plt.grid(True)
        if show: plt.show()
        if figname != None: plt.savefig('Calibration_data.pdf')

    def filter_peaks(self, threshold: int = None):
        """
        Filter the data by supressing all the data above the given threshold.

        Parameters
        ----------
        threshold:
            Default: None. Threshold under which data are not considered.

        Returns
        -------
        Data
            The filtered data.
        """
        if threshold != None: self.threshold = threshold
        assert self.threshold != None, 'Error: No treshold specified for the filter_peaks method.'
        filtered_data = self.data.copy() #deep copy of the data
        if self.threshold:
            filter_array = np.where(self.data[:, 1] > self.threshold, 1, 0)
            filtered_data[:, 1] = filtered_data[:, 1] * filter_array
        filtered_data = Data(filtered_data) #create a new instance for the filtered data
        return filtered_data

    def compute_stats(self, interval_start: int, interval_end: int):
        """
        Find the bin with the most hits and returns the bin index, number of hits and the standard deviation.

        Parameters
        ----------
        interval_start: int
            Beginning of the interval in which we compute the stats.
        interval_end: int
            End of the interval in which we compute the stats.

        Returns
        -------
        tuple
            Tuple containing the bins, the number of hits and the mean quadratic deviation to the nominal bin in bin units.
        """
        nominal_bin, nb_counts = max(self.data[interval_start:interval_end, :], key=lambda x: x[1]) #get the max

        variance = 0 #initialize the square of the mean quadratic deviation (not a true variance)
        normalization_cst = 0 #initialize the normalization constant
        for hits in self.data[interval_start:interval_end, 1]:
            if hits:
                variance += (nb_counts - hits) ** 2
                normalization_cst += 1
        variance /= normalization_cst

        return nominal_bin, nb_counts, np.sqrt(variance)

    def calibrate(self, interval_list: list, peak_list: list, display: bool = False, plot: bool = False):
        """
        Perform the calibration.

        Parameters
        ----------
        interval_list: list
            List of tuples (start, end) delimiting all calibration peaks.
        peak_list: list
            List of known energies to match to the observed peaks.
        display: bool, optional
            Default: False. If True, displays calibration results.
        plot: bool, optional
            Default: False. If True, displays a plot of the calibration data.

        Returns
        -------
        tuple
            Tuple containing: the function mapping QDC channels to energies, the minimum quadratic deviation to the nominal energy, and the R² coefficient.
        """
        # filter data and get the peaks
        nominal_bins = [] #list of nominal bins, for each peak
        quadratic_deviations = [] #list of mean quadratic deviations from nominal bin, for each peak
        filtered_data = self.filter_peaks()
        for idx_start, idx_end in interval_list:
            nominal_bin, nb_counts, qd = filtered_data.compute_stats(idx_start,idx_end)
            nominal_bins.append([nominal_bin])
            quadratic_deviations.append(qd)
            if display:
                print(
                    f"Bin number {nominal_bin}\n"
                    f".. number of counts         = {nb_counts}\n"
                    f".. mean quadratic deviation = {qd:.3f}"
                )

        if plot: self.plot_data() #plot the data and show the filter

        # perform the calibration
        nominal_bins = np.array(nominal_bins)
        quadratic_dev = min(quadratic_deviations)
        nominal_energies = np.array(peak_list) #known energies in MeV

        # linear regression
        model = LinearRegression()
        model.fit(nominal_bins, nominal_energies)
        pred_energies = model.predict(nominal_bins) #predict using the model
        r2 = r2_score(nominal_energies, pred_energies) #calculate the R² score
        slope = model.coef_[0] #slope of the model
        intercept = model.intercept_ #intercept of the model
        convert = lambda x: (slope * x + intercept)

        quadratic_dev = convert(quadratic_dev) #convert bins into MeV

        if display:
            print("Calibration results\n"
                f".. slope       = {slope} MeV\n"
                f".. intercept   = {intercept} MeV\n"
                f".. uncertainty = {quadratic_dev} MeV\n"
                f".. R²          = {r2}"
            )

        return convert, quadratic_dev, r2

    def compute_experimental_number_of_counts(self, interval_start: int, interval_end: int, mean: float, std_dev: float, amplitude: float, slope: float, intercept: float, display: bool = False):
        """
        Compute experimental number of counts in the detector at a certain angle, by fitting a Gaussian distribution to each peak and subtracting the noise described by an affine fit.

        Parameters
        ----------
        interval_start: int
            Start of the interval delimiting the peak we have to fit.
        interval_end: int
            End of the interval delimiting the peak we have to fit.
        mean:float
            Initial guess for the mean of the Gaussian distribution.
        std_dev:float
            Initial guess for the standard deviation of the Gaussian distribution.
        amplitude:float
            Initial guess for the amplitude of the Gaussian distribution.
        slope: float
            Initial guess for the slope of the noise.
        intercept: float
            Initial guess for the intercept of the noise.
        display: boolean, optional
            Default: False. If True, displays the results of the computation.

        Returns
        -------
        int
            Experimental number of counts corresponding to detected particles.
        callable
            Fitted function.
        """
        # fit the histogram
        fit_function = lambda x, mean, std_dev, amplitude, slope, intercept: (amplitude * np.exp(-0.5 * ((x - mean) / std_dev) ** 2) + slope * x + intercept) #function we want to fit to the data
        initial_guess = [mean, std_dev, amplitude, slope, intercept] #initial guess for the parameters
        params, covariance = curve_fit(fit_function, self.data[interval_start:interval_end, 0], self.data[interval_start:interval_end, 1], p0=initial_guess) #fit the Gaussian function to the histogram data
        fitted_mean, fitted_std_dev, fitted_amplitude, fitted_slope, fitted_intercept = params

        # compute the number of counts
        nb_counts = int(fitted_amplitude * fitted_std_dev * np.sqrt(2*np.pi))
        noise = fitted_slope/2 * (interval_end**2 - interval_start**2) + fitted_intercept * (interval_end - interval_start)
        #nb_counts = sum(self.data[interval_start:interval_end, 1]) - noise
        fitted_function = lambda x: fit_function(x, fitted_mean, fitted_std_dev, fitted_amplitude, fitted_slope, fitted_intercept)

        if display:
            print(
                'fitted mean      = ', fitted_mean, '\n'
                'fitted std dev   = ', fitted_std_dev, '\n'
                'fitted amplitude = ', fitted_amplitude, '\n'
                'fitted slope     = ', fitted_slope, '\n'
                'fitted intercept = ', fitted_intercept, '\n'
                'noise            = ', noise, '\n'
                'number of counts = ', nb_counts
            )

        return nb_counts, fitted_function

    def compute_experimental_cross_section(self, collision, nb_counts: int):
        """
        Compute experimental cross-section.

        Parameters
        ----------
        collision: Collision
            The experiment under consideration.
        nb_counts: int
            Number of counts in the detector corresponding to scattered nuclei.

        Returns
        -------
        float
            Experimental differential cross-section in mb/sr.
        """
        cross_section = nb_counts / (collision.nb_particles * collision.target_density * collision.detector_solid_angle) * 1e31
        return cross_section





