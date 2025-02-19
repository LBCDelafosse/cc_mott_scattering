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
    # set the conditions of the experiment
    theta_start = 5 * np.pi/180 #initial angle for plot in radians
    theta_stop = 175* np.pi/180 #final angle for plot in radians
    angles = np.linspace(theta_start, theta_stop, 10000)
    Z1,A1 = 6,12
    E_beam = 3. #lab frame energy of the beam particles

    # initialize lists of cross-sections
    spin0_cross_sections = []
    spin1_cross_sections = []
    spin2_cross_sections = []

    # compute Mott cross-sections
    for angle in angles:
        spin0_collision = Collision(angle, 0, Z1, A1, Z1, A1, E_beam)
        #spin0_collision.set_detector_angle_from_COM_angle(angle)
        spin0_cross_sections.append(spin0_collision.compute_mott_cross_section(0, display=True))
        spin1_collision = Collision(angle, 0, Z1, A1, Z1, A1, E_beam)
        #spin1_collision.set_detector_angle_from_COM_angle(angle)
        spin1_cross_sections.append(spin1_collision.compute_mott_cross_section(1))
        spin2_collision = Collision(angle, 0, Z1, A1, Z1, A1, E_beam)
        #spin2_collision.set_detector_angle_from_COM_angle(angle)
        spin2_cross_sections.append(spin2_collision.compute_mott_cross_section(2))

    # plot
    xlabel = 'Scattering angle (°)'
    ylabel = 'Cross-section (mb/sr)'
    angles *= 180/np.pi #conversion in degrees
    dt.plot([angles for _ in range(3)], [spin0_cross_sections, spin1_cross_sections, spin2_cross_sections], ['k-','b--','r:'], ['I=0','I=1','I=2'], xlabel, ylabel, log=True, show=True)

def calibration():
    data = create_data_from_filename("calibration_data.asc")
    data.set_threshold(15) #set a threshold to filter the data
    interval_list = [(12500, 13000), (13500, 14000), (14000, 14500)]
    peak_list = [5.16, 5.486, 5.805]
    convert, std_dev, r2 = data.calibrate(interval_list, peak_list, display=True)

def theoretical_nb_counts():
    # experiment parameters
    detector_angle = 90 * np.pi/180 #angle of the detector in the lab frame in radians
    detector_solid_angle = 2*np.pi * (1. - 1. / np.sqrt(1. + (0.5/130.)**2)) #solid angle occupied by the detector
    Z1,A1 = 1,1
    E_beam = 3. #lab frame energy in MeV
    au_collision = Collision(detector_angle, detector_solid_angle, Z1, A1, 79, 197, E_beam)
    au_collision.set_nb_particles_from_faraday(10149)
    au_collision.set_target_density_from_mass_density(80.)
    c_collision = Collision(detector_angle, detector_solid_angle, Z1, A1, 6, 12, E_beam)
    c_collision.set_nb_particles_from_faraday(10149)
    c_collision.set_target_density_from_mass_density(10.)

    # compute theoretical number of counts
    au_detected_energy, au_nb_counts = au_collision.compute_number_of_counts()
    c_detected_energy, c_nb_counts = c_collision.compute_number_of_counts()

    print(
        f"Theoretical number of counts at {detector_angle}°\n"
        f"..Au: {au_nb_counts}\n"
        f"..C : {c_nb_counts}\n"
    )

def fit_peak():
    # set the conditions of the experiment
    detector_solid_angle = 2*np.pi * (1. - 1. / np.sqrt(1. + (0.5/123.)**2)) #solid angle occupied by the detector
    Z1,A1,Z2,A2 = 1,1,79,197
    E_beam = 3. #lab frame energy of the beam particles
    collision = Collision(np.pi/2, detector_solid_angle, Z1, A1, Z2, A2, E_beam)
    collision.set_nb_particles_from_faraday(10149)
    collision.set_target_density_from_mass_density(80.)

    # compute theoretical cross-section
    theoretical = collision.compute_rutherford_cross_section()

    # extract experimental data from the file
    data = create_data_from_filename("Exp/Beam2_90Deg_PosA/CardStella_0122_histo_V1.asc")

    # compute experimental cross-section
    nb_counts, fitted_function = data.compute_experimental_number_of_counts(6593, 10000, 7.2e3, 50., 50., 0., 2., display=True) #retrieve experimental number of counts and number of counts per bin
    cross_section = data.compute_experimental_cross_section(collision, nb_counts) #compute cross-section

    y_fit = fitted_function(data[:, 0])
    print(
        f"experimental cross-section = {cross_section} mb/sr\n"
        f"theoretical cross-section  = {theoretical} mb/sr"
    )

    # plot
    plt.plot(data[:, 0], data[:, 1], color='C0')
    plt.plot(data[:, 0], y_fit, 'r-', label='Fitted Curve')
    plt.xlabel('Bins')
    plt.ylabel('Number of counts')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    #plot_carbon_gold_cross_section()
    #plot_mott_section() # PROBLEM !!!!!!!!!!
    calibration() # PROBLEM !!!!!!!!!!!!
    #theoretical_nb_counts()
    #fit_peak()






