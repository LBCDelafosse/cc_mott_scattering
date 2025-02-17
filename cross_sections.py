import numpy as np
import scipy.constants as cst
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from collision import *
import data_extraction as dt


def rutherford_cross_sections(angles, A1, Z1, A2, Z2, E_beam):
    """
    Computes theoretical Rutherford cross-sections for a given collision.

    Parameters
    ----------
    angles: numpy array
        List of angles at which measurements are done in the COM frame, in radians.
    A1: int
        Mass number of the beam particle.
    Z1: int
        Charge number of the beam particle.
    A2: int
        Mass number of the target particle.
    Z2: int
        Charge number of the target particle.
    E_beam: float
        Lab frame beam energy in MeV.

    Returns
    -------
    cross_sections: numpy array
        List of theoretical Rutherford cross-sections, for each angle in the COM frame, in mb.
    """
    E_COM = E_beam * A2 / (A1+A2) #center-of-mass energy in MeV
    nb_entries = len(angles)
    mu = cst.m_n * A1 * A2 / (A1 + A2)
    initial_v = np.sqrt(E_COM*(10**6)*cst.e*2/mu)
    k = mu * initial_v / cst.hbar
    n = (Z1*Z2 * cst.e*cst.e) / (4*np.pi*cst.epsilon_0 * initial_v * cst.hbar)

    cross_sections = np.zeros(nb_entries)

    for index in range(nb_entries):
        theta = angles[index]
        cross_sections[index] = n*n / (4.*k*k) * cosec(theta/2.)**4 #SI units
#        if 'absorption' in plot_instructions and A1 != A2:
#            absorption_factor = np.exp(-np.abs(A1 - A2) / (A1 + A2))
#            cross_sections[index] *= absorption_factor
        cross_sections[index] *= 10**31 #conversion in mbarns

    return cross_sections

def mott_cross_sections(angles, A1, Z1, A2, Z2, E_beam, total_spin):
    """
    Computes theoretical Mott cross-sections for a given collision.

    Parameters
    ----------
    angles: numpy array
        List of angles at which measurements are done in COM frame, in radians.
    A1: int
        Mass number of the beam particle.
    Z1: int
        Charge number of the beam particle.
    A2: int
        Mass number of the target particle.
    Z2: int
        Charge number of the target particle.
    E_beam: float
        Lab frame beam energy in MeV.
    total_spin: float
        Total spin of the target/beam atom when they are indistinguishable.

    Returns
    -------
    cross_sections: numpy array
        List of theoretical Mott cross-sections, for each angle in the COM frame, in mb.
    """
    E_COM = E_beam * A2 / (A1+A2) #center-of-mass energy in MeV
    nb_entries = len(angles)
    mu = cst.m_n * A1 * A2 / (A1 + A2)
    initial_v = np.sqrt(E_COM*(10**6)*cst.e*2/mu)
    k = mu * initial_v / cst.hbar
    n = (Z1*Z2 * cst.e*cst.e) / (4*np.pi*cst.epsilon_0 * initial_v * cst.hbar)

    cross_sections = np.zeros(nb_entries)

    for index in range(nb_entries):
        theta = angles[index]
        cross_sections[index] = n*n / (4.*k*k) * ( cosec(theta/2.)**4 + sec(theta/2.)**4 + 2*(-1)**(2*total_spin)/(2*total_spin+1) * np.cos(n * np.log(np.tan(theta/2.)**2)) * cosec(theta/2.)**2 * sec(theta/2.)**2 ) #SI units
        cross_sections[index] *= 10**31 #conversion in mbarns

    return cross_sections

def plot(xaxes, yaxes, linetypes, labels, xlabel, ylabel, log=False, show=True, figname=None):
    """
    Plots a graph.

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

def number_of_counts(detector_angle, det_solid_angle, particle_nb, target_density, A1, Z1, A2, Z2, E_COM, display=False):
    """
    Computes the theoretical number of counts for the detected energy (Rutherford scattering).

    Parameters
    ----------
    detector angle: float
        Angle at which the detector is placed in the lab frame, in radians.
    det_solid_angle: float
        Solid angle occupied by the detector in the lab frame, in steradians.
    particle_nb: int
        Number of particles in the incident beam.
    target_density: float
        Number of scattering centres per squared meter.
    A1: int
        Mass number of the beam particle.
    Z1: int
        Charge number of the beam particle.
    A2: int
        Mass number of the target particle.
    Z2: int
        Charge number of the target particle.
    E_COM: float
        Center-of-mass energy in MeV.
    display: boolean
        If True, displays a number of calculation results.

    Returns
    -------
    det_energy: float
        Detected energy in MeV.
    nb_counts: int
        Number of counts.
    """
    gamma = A1/A2
    conversion_numerator = gamma * np.cos(detector_angle) - np.sqrt(1. - gamma**2*np.sin(detector_angle)**2)
    det_energy = A1**2 * E_COM / (A2*(A1+A2)) * conversion_numerator**2 / gamma**2 #detected energy
    theta_COM = np.arcsin( np.sin(detector_angle) * conversion_numerator ) #angle in the center-of-mass frame
    if display: print(theta_COM * 180 / np.pi, '°')
    E_beam = E_COM * (A1+A2) / A2 #lab frame beam energy
    [cross_section_COM] = rutherford_cross_sections([theta_COM], A1, Z1, A2, Z2, E_beam)
    if display: print('cross-section COM =', cross_section_COM, 'mb/sr')
    cross_section = cross_section_COM * conversion_numerator**2 / np.sqrt(1. - gamma**2 * np.sin(detector_angle)**2) #take the cross-section to the lab frame, in mb
    if display: print('cross-section =', cross_section, 'mb/sr')
    nb_counts = det_solid_angle * cross_section * 10**(-31) * particle_nb * target_density
    if display: print('det_energy =', det_energy, 'MeV, nb_counts =', nb_counts)
    return det_energy, nb_counts

if __name__ == "__main__":
    theta_start = 5 * np.pi/180 #initial angle for plot in radians
    theta_stop = 175* np.pi/180 #final angle for plot in radians
    angles = np.linspace(theta_start, theta_stop, 10000)
    E_beam = 3. #lab frame energy of the beam particles

    #A1,Z1,A2,Z2 = 1,1,12,6 #collision on carbon
    #c_yaxis = rutherford_cross_sections(angles, A1, Z1, A2, Z2, E_beam)
    A1,Z1,A2,Z2 = 1,1,197,79 #collision on gold
    #au_yaxis = rutherford_cross_sections(angles, A1, Z1, A2, Z2, E_beam)
    [value] = rutherford_cross_sections([2*np.pi/3], A1, Z1, A2, Z2, E_beam)
    print('value =', value)

    #spin0_yaxis = mott_cross_sections(angles, A1, Z1, A2, Z2, E_beam, 0)
    #spin1_yaxis = mott_cross_sections(angles, A1, Z1, A2, Z2, E_beam, 1)
    #spin2_yaxis = mott_cross_sections(angles, A1, Z1, A2, Z2, E_beam, 2)

    xlabel = 'Scattering angle (°)'
    ylabel = 'Cross-section (mb/sr)'
    angles *= 180/np.pi #conversion in degrees
    #plot([angles for _ in range(2)], [au_yaxis, c_yaxis], ['r--','k--'], ['Gold','Carbon'], xlabel, ylabel, log=True)
    #plot([angles for _ in range(3)], [spin0_yaxis, spin1_yaxis, spin2_yaxis], ['k-','b--','r:'], ['I=0','I=1','I=2'], xlabel, ylabel, log=True)


