import numpy as np
import scipy.constants as cst
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit


def sec(x):
    """
    Compute the trigonometric secant function.
    """
    return 1./np.cos(x)

def cosec(x):
    """
    Compute the trigonometric cosecant function.
    """
    return 1./np.sin(x)

def reciprocal_sin(x):
    """
    Compute the reciprocal sine.
    """
    floor = int(x/np.pi + 0.5)
    return floor

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

    Returns
    -------
    Nothing.
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

class Collision:
    """
    Object containing the parameters of an elastic collision in a collision experiment.
    """

    # Constructors

    def __init__(self, detector_angle, detector_solid_angle, Z1, A1, Z2, A2, E_beam, nb_particles=0, target_density=0., total_spin=0.):
        """
        Constructor of the Collision class.

        Parameters
        ----------
        detector_angle: float
            Angle at which the detector is placed in the lab frame, in radians.
        detector_solid_angle: float
            Solid angle occupied by the detector in the lab frame, in steradians.
        Z1: int
            Charge number of the beam particle.
        A1: int
            Mass number of the beam particle.
        Z2: int
            Charge number of the target particle.
        A2: int
            Mass number of the target particle.
        E_beam: float
            Energy of beam particles in MeV.
        nb_particles: int, optional
            Default: 0. Number of particles in the incident beam.
        target_density: float, optional
            Default: 0. Number of scattering centres per squared meter.
        total_spin: float, optional
            Default: 0. Total spin of the target/beam atom when they are indistinguishable.
        """
        self.detector_angle = detector_angle
        self.detector_solid_angle = detector_solid_angle
        self.Z1 = Z1
        self.A1 = A1
        self.Z2 = Z2
        self.A2 = A2
        self.E_beam = E_beam
        self.nb_particles = nb_particles
        self.target_density = target_density
        self.total_spin = total_spin
        reduced_mass = cst.m_n * self.A1 * self.A2 / (self.A1 + self.A2)
        initial_velocity = np.sqrt(self.E_beam * 1e6 * cst.e * 2 / (self.A1 * cst.m_n))
        self.__k = reduced_mass * initial_velocity / cst.hbar
        self.__n = (self.Z1 * self.Z2 * cst.e**2) / (4 * np.pi * cst.epsilon_0 * initial_velocity * cst.hbar)

    def copy(self):
        """
        Construct a copy of the instance.

        Returns
        -------
        Collision
            A copy of the instance.
        """
        return Collision(self.detector_angle, self.detector_solid_angle, self.Z1, self.A1, self.Z2, self.A2, self.E_beam, self.nb_particles, self.target_density, self.total_spin)

    # Accessors

    def get_kinematical_parameters(self):
        """
        Returns
        -------
        tuple
            Tuple containing: the de Broglie wavelength in SI units, and the kinemtical number n.
        """
        return self.__k, self.__n

    # Modifiers

    def set_detector_angle_from_COM_angle(self, COM_angle):
        gamma, conversion_constant = self._compute_conversion_constants(-1)
        self.detector_angle = np.arctan( np.sin(COM_angle) / (np.cos(COM_angle) + gamma) )
        if self.detector_angle < 0: self.detector_angle += np.pi
        return self.detector_angle

    def set_nb_particles_from_intensity(self, beam_intensity, duration):
        """
        Set the instance attribute nb_particles from the beam intensity and experiment duration.

        Parameters
        ----------
        beam_intensity: float
            Beam intensity in pnA.
        duration:
            Duration of the experiment in seconds.

        Returns
        -------
        int
            Number of incident particles in the beam.
        """
        self.nb_particles = int(duration * beam_intensity * 6.25e9)
        return self.nb_particles

    def set_nb_particles_from_faraday(self, faraday_count):
        """
        Set the instance attribute nb_particles from the number of counts in the Faraday cup.

        Parameters
        ----------
        faraday_count: int
            Number of counts in the Faraday cup.

        Returns
        -------
        int
            Number of incident particles in the beam.
        """
        self.nb_particles = faraday_count * 2e-9 / cst.e
        return self.nb_particles

    def set_target_density_from_mass_density(self, mass_density):
        """
        Set the instance attribute target_density from the mass density.

        Parameters
        ----------
        mass_density: float
            Surface mass density, in µg/cm².

        Returns
        -------
        float
            Number of scattering centres per squared meter in the target.
        """
        self.target_density = mass_density * 1e-2 * cst.N_A / self.A2
        return self.target_density

    # Other methods

    def compute_mass_and_velocity(self):
        """
        Compute the reduced mass and the initial velocity of the incident particles in the lab frame.

        Returns
        -------
        tuple
            Tuple containing: the system's reduced mass in kg, the lab frame initial velocity of the incident particles in m/s.
        """
        reduced_mass = cst.m_n * self.A1 * self.A2 / (self.A1 + self.A2)
        initial_velocity = np.sqrt(self.E_beam * 1e6 * cst.e * 2 / (self.A1 * cst.m_n))
        return reduced_mass, initial_velocity

    def _compute_conversion_constants(self, sign):
        gamma = self.A1 / self.A2
        conversion_constant = gamma * np.cos(self.detector_angle) + sign * np.sqrt(1. - gamma**2 * np.sin(self.detector_angle)**2)
        return gamma, conversion_constant

    def compute_E_COM(self):
        """
        Compute the total system's energy in the centre-of-mass frame.

        Returns
        -------
        float
            Energy of the total system in the centre-of-mass frame, in MeV.
        """
        return E_beam * self.A2 / (self.A1 + self.A2)

    def compute_COM_angle(self):
        """
        Compute the detector angle in the COM frame.

        Returns
        -------
        float
            Detector angle in the COM frame, in radians.
        """
        gamma, conversion_constant = self._compute_conversion_constants(+1)
        COM_angle = np.arcsin( np.sin(self.detector_angle) * conversion_constant )
        if self.detector_angle > np.pi/2:
            COM_angle = np.pi - COM_angle
        return COM_angle

    def compute_detected_energy(self):
        """
        Compute the energy deposited by each scattered particle in the detector.

        Returns
        -------
        float
            Detector angle in the COM frame, in radians.
        """
        gamma, conversion_constant = self._compute_conversion_constants(-1)
        return self.A1**2 * E_beam / (self.A1 + self.A2)**2 * conversion_constant**2 / gamma**2

    def compute_rutherford_cross_section(self):
        """
        Compute theoretical Rutherford cross-section for a given collision.

        Returns
        -------
        float
            Theoretical Rutherford cross-section in the centre-of-mass frame, in mb/sr.
        """
        COM_angle = self.compute_COM_angle()
        return self.__n**2 / (4. * self.__k**2) * cosec(COM_angle/2.)**4 * 1e31

    def compute_mott_cross_section(self):
        """
        Compute theoretical Mott cross-section for a given collision.

        Returns
        -------
        float
            Theoretical Mott cross-section in the centre-of-mass, in mb/sr.
        """
        COM_angle = self.compute_COM_angle()
        return self.__n**2 / (4. * self.__k**2) * ( cosec(COM_angle/2.)**4 + sec(COM_angle/2.)**4 + 2*(-1)**(2*self.total_spin)/(2*self.total_spin+1) * np.cos(self.__n * np.log(np.tan(COM_angle/2.)**2)) * cosec(COM_angle/2.)**2 * sec(COM_angle/2.)**2 ) * 1e31

    def compute_number_of_counts(self, display=False):
        """
        Compute the theoretical number of counts for the detected energy (Rutherford scattering).

        Returns
        -------
        tuple
            Tuple containing: the detected energy in MeV, the theoretical number of counts in the detector.
        """
        gamma, conversion_constant = self._compute_conversion_constants(-1)

        detected_energy = self.A1**2 * self.E_beam / (self.A1 + self.A2)**2 * conversion_constant**2 / gamma**2 #energy deposited by each 
        cross_section = self.compute_rutherford_cross_section() * conversion_numerator**2 / np.sqrt(1. - gamma**2 * np.sin(self.detector_angle)**2) #take the cross-section to the lab frame, in mb/sr
        nb_counts = self.detector_solid_angle * cross_section * 1e-31 * self.nb_particles * self.target_density

        if display:
            print('det_energy =    ', detected_energy, 'MeV')
            print('cross-section = ', cross_section, 'mb/sr')
            print('nb_counts =     ', nb_counts)

        return detected_energy, nb_counts


if __name__ == "__main__":
    theta_start = 5 * np.pi/180 #initial angle for plot in radians
    theta_stop = 175* np.pi/180 #final angle for plot in radians
    angles = np.linspace(theta_start, theta_stop, 10000)
    E_beam = 3. #lab frame energy of the beam particles

    A1,Z1,A2,Z2 = 1,1,197,79 #collision on gold

    cross_sections = []

    for angle in angles:
        au_collision = Collision(0, 0, Z1, A1, Z2, A2, E_beam)
        au_collision.set_detector_angle_from_COM_angle(angle)
        cross_sections.append(au_collision.compute_rutherford_cross_section())

    xlabel = 'Scattering angle (°)'
    ylabel = 'Cross-section (mb/sr)'
    angles *= 180/np.pi #conversion in degrees
    plot([angles], [cross_sections], ['r-'], ['Gold'], xlabel, ylabel, log=True)












