import numpy as np
import scipy.constants as cst
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from collision import *
from data import *
import data_extraction as dt


def plot_carbon_gold_cross_section():
    theta_start = 5 * np.pi/180 #initial angle for plot in radians
    theta_stop = 175* np.pi/180 #final angle for plot in radians
    angles = np.linspace(theta_start, theta_stop, 10000)
    Z1,A1 = 1,1
    E_beam = 3. #lab frame energy of the beam particles

    au_cross_sections = []
    c_cross_sections = []

    for angle in angles:
        au_collision = Collision(angle, 0, Z1, A1, 79, 197, E_beam)
        au_cross_sections.append(au_collision.compute_rutherford_cross_section())
        c_collision = Collision(angle, 0, Z1, A1, 6, 12, E_beam)
        c_cross_sections.append(c_collision.compute_rutherford_cross_section())

    xlabel = 'Scattering angle (°)'
    ylabel = 'Cross-section (mb/sr)'
    angles *= 180/np.pi #conversion in degrees
    dt.plot([angles for _ in range(2)], [au_cross_sections, c_cross_sections], ['r--', 'k--'], ['Gold', 'Carbon'], xlabel, ylabel, log=True, show=True)

def plot_mott_section():
    theta_start = 5 * np.pi/180 #initial angle for plot in radians
    theta_stop = 175* np.pi/180 #final angle for plot in radians
    angles = np.linspace(theta_start, theta_stop, 10000)
    Z1,A1 = 6,12
    E_beam = 3. #lab frame energy of the beam particles

    spin0_cross_sections = []
    spin1_cross_sections = []
    spin2_cross_sections = []

    for angle in angles:
        spin0_collision = Collision(angle, 0, Z1, A1, Z1, A1, E_beam)
        spin0_collision.set_detector_angle_from_COM_angle(angle)
        spin0_cross_sections.append(spin0_collision.compute_mott_cross_section(0))
        spin1_collision = Collision(angle, 0, Z1, A1, Z1, A1, E_beam)
        spin1_collision.set_detector_angle_from_COM_angle(angle)
        spin1_cross_sections.append(spin1_collision.compute_mott_cross_section(1))
        spin2_collision = Collision(angle, 0, Z1, A1, Z1, A1, E_beam)
        spin2_collision.set_detector_angle_from_COM_angle(angle)
        spin2_cross_sections.append(spin2_collision.compute_mott_cross_section(2))

    xlabel = 'Scattering angle (°)'
    ylabel = 'Cross-section (mb/sr)'
    angles *= 180/np.pi #conversion in degrees
    dt.plot([angles for _ in range(3)], [spin0_cross_sections, spin1_cross_sections, spin2_cross_sections], ['k-','b--','r:'], ['I=0','I=1','I=2'], xlabel, ylabel, log=True, show=True)

def fit_peak():
    data = create_data_from_filename("Exp/Beam2_90Deg_PosA/CardStella_0122_histo_V1.asc")
    
    detector_solid_angle = 2*np.pi * (1. - 1. / np.sqrt(1. + (0.5/123.)**2)) #solid angle occupied by the detector
    Z1,A1,Z2,A2 = 1,1,79,197
    E_beam = 3. #lab frame energy of the beam particles
    collision = Collision(np.pi/2, detector_solid_angle, Z1, A1, Z2, A2, E_beam)
    collision.set_nb_particles_from_faraday(10149)
    collision.set_target_density_from_mass_density(80.)
    theoretical = collision.compute_rutherford_cross_section()

    nb_counts, fitted_function = data.compute_experimental_number_of_counts(6593, 10000, 7.2e3, 50., 50., 0., 2.)
    cross_section = data.compute_experimental_cross_section(collision, nb_counts)

    y_fit = fitted_function(data[:, 0])
    print(nb_counts)
    print(cross_section)
    print(theoretical)

    plt.plot(data[:, 0], data[:, 1], color='C0')
    plt.plot(data[:, 0], y_fit, 'r-', label='Fitted Curve')

    # Add labels and legend
    plt.xlabel('Bins')
    plt.ylabel('Number of counts')
    plt.legend()

    # Show the plot
    plt.show()
    #plt.savefig('../Fit.pdf')

if __name__ == "__main__":
    #plot_carbon_gold_cross_section()
    #plot_mott_section()
    fit_peak()



#peak_list = [(12500, 13000), (13500, 14000), (14000, 14500)]





