import numpy as np
import scipy.constants as cst
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from collision import *
from data import *
from functions import *


def plot_carbon_gold_cross_section():
    """
    Plot the Rutherford cross-sections for proton on Gold 197 and Carbon 12, in the centre-of-mass frame.
    """
    theta_start = 5 * np.pi/180 #initial angle for plot in radians
    theta_stop = 175* np.pi/180 #final angle for plot in radians
    angles = np.linspace(theta_start, theta_stop, 100)
    Z1,A1 = 1,1
    E_beam = 3. #lab frame energy of the beam particles

    au_cross_sections = []
    c_cross_sections = []

    for angle in angles:
        au_collision = Collision(angle, 0, Z1, A1, 79, 197, E_beam)
        au_collision.set_detector_angle_from_COM_angle(angle)
        au_cross_sections.append(au_collision.compute_rutherford_cross_section())
        c_collision = Collision(angle, 0, Z1, A1, 6, 12, E_beam)
        c_collision.set_detector_angle_from_COM_angle(angle)
        c_cross_sections.append(c_collision.compute_rutherford_cross_section())

    xlabel = 'Scattering angle (°)'
    ylabel = 'Cross-section (mb/sr)'
    angles *= 180/np.pi #conversion in degrees
    plot([angles for _ in range(2)], [au_cross_sections, c_cross_sections], ['r--', 'k--'], ['Gold', 'Carbon'], xlabel, ylabel, log=True, show=True)

def plot_mott_section():
    """
    Plot the Mott cross-sections for Carbon 12 - Carbon 12 scattering, in the centre-of-mass frame, for different spin values.
    """
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
        spin0_collision.set_detector_angle_from_COM_angle(angle)
        spin0_cross_sections.append(spin0_collision.compute_mott_cross_section(0))
        spin1_collision = Collision(angle, 0, Z1, A1, Z1, A1, E_beam)
        spin1_collision.set_detector_angle_from_COM_angle(angle)
        spin1_cross_sections.append(spin1_collision.compute_mott_cross_section(1))
        spin2_collision = Collision(angle, 0, Z1, A1, Z1, A1, E_beam)
        spin2_collision.set_detector_angle_from_COM_angle(angle)
        spin2_cross_sections.append(spin2_collision.compute_mott_cross_section(2))

    # plot
    xlabel = 'Scattering angle (°)'
    ylabel = 'Cross-section (mb/sr)'
    angles *= 180/np.pi #conversion in degrees
    plot([angles for _ in range(3)], [spin0_cross_sections, spin1_cross_sections, spin2_cross_sections], ['k-','b--','r:'], ['I=0','I=1','I=2'], xlabel, ylabel, log=True, show=True)

def calibration():
    """
    Perform the calibration mapping QDC channels on energies.
    """
    data = create_data_from_filename("calibration_data.asc")
    data.set_threshold(15) #set a threshold to filter the data
    interval_list = [(12500, 13000), (13500, 14000), (14000, 14500)] #bin intervals in which the peaks must be searched
    peak_list = [5.16, 5.486, 5.805] #list of known energies to which the peaks must be matched
    convert, std_dev, r2 = data.calibrate(interval_list, peak_list, display=True)

def theoretical_nb_counts():
    """
    Compute the theoretical total number of scattered nuclei and their energies.
    """
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
        f"Theoretical number of counts at {detector_angle * 180/np.pi}°\n"
        f"..Au: {au_nb_counts} counts at {au_detected_energy} MeV\n"
        f"..C : {c_nb_counts} counts at {c_detected_energy} MeV\n"
    )

def fit_peak():
    """
    Illustrates the computation of an experimental cross-section for proton on Gold 197 scattering, at 90°.
    """
    # experiment parameters
    angle = 90 * np.pi / 180
    detector_solid_angle = 2*np.pi * (1. - 1. / np.sqrt(1. + (0.5/123.)**2)) #solid angle occupied by the detector
    Z1,A1,Z2,A2 = 1,1,79,197
    E_beam = 3. #lab frame energy of the beam particles
    faraday_count = 10149

    collision = Collision(angle, detector_solid_angle, Z1, A1, Z2, A2, E_beam)
    collision.set_nb_particles_from_faraday(faraday_count)
    collision.set_target_density_from_mass_density(80.)

    # compute theoretical cross-section
    theoretical = collision.compute_lab_rutherford_cross_section()

    # extract experimental data from the file
    data = create_data_from_filename("Exp/Beam2_90Deg_PosA/CardStella_0122_histo_V1.asc")

    # compute experimental cross-section
    nb_counts, fitted_function = data.compute_experimental_number_of_counts(6200, 7500, 7.2e3, 50., 50., 0., 2., display=True) #retrieve experimental number of counts and number of counts per bin
    cross_section = data.compute_experimental_cross_section(collision, nb_counts) #compute cross-section

    y_fit = fitted_function(data[:, 0])
    print(
        f"experimental cross-section = {cross_section} mb/sr\n"
        f"theoretical cross-section  = {theoretical} mb/sr"
    )

    # plot
    plt.rcParams.update({
        "font.family": "serif",   # Type of font
        "xtick.labelsize": "20",  # Size of the xtick label
        "ytick.labelsize": "20",  # Size of the ytick label
        "lines.linewidth": "2",   # Width of the curves
        "legend.framealpha": "1", # Transparency of the legend frame
        "legend.fontsize": "23",  # Size of the legend
        'axes.labelsize' : '23',  # Size of the axes' labels
        "grid.linestyle":"--",    # Grid formed by dashed lines
        "text.usetex": True       # Using LaTex style for text and equation
    })

    plt.plot(data[:, 0], data[:, 1], color='C0')
    plt.plot(data[:, 0], y_fit, 'r-', label='Fitted Curve')
    plt.xlabel('Bins')
    plt.ylabel('Number of counts')
    plt.legend()
    plt.tight_layout()
    plt.show()

def get_experimental_cross_sections():
    """
    Compare the experimental and theoretical cross-sections for proton on Gold 197 scattering.
    """
    # define lists of angles
    # each list corresponds to a different position of the target
    angles_A = np.array([60, 70, 80, 90])
    angles_B = np.array([90, 100, 110, 120, 130])
    nb_files_A = len(angles_A)
    nb_files_B = len(angles_B)

    # define number of counts
    faraday_counts_A = np.array([3206, 4206, 4595, 10149])
    faraday_counts_B = np.array([8472, 9714, 12563, 14391, 17002, 20078])

    # list of intervals for searching the Au peak
    intervals_A = [
        (4900, 7500),
        (5000, 7500),
        (5000, 7500),
        (6200, 7500)
    ]
    intervals_B = [
        (6100, 7400),
        (5900, 7400),
        (5800, 7400),
        (5500, 7170),
        (5500, 7170),
        (5500, 7170)
    ]
    
    # lists of data files
    file_list_A = np.array([
        "Exp/Beam5_60Deg_PosA/CardStella_0122_histo_V1.asc",
        "Exp/Beam4_70Deg_PosA/CardStella_0122_histo_V1.asc",
        "Exp/Beam3_80Deg_PosA/CardStella_0122_histo_V1.asc",
        "Exp/Beam2_90Deg_PosA/CardStella_0122_histo_V1.asc"
    ])
    file_list_B = np.array([
        "Exp/Beam6_90Deg_PosB/CardStella_0122_histo_V1.asc",
        "Exp/Beam7_100Deg_PosB/CardStella_0122_histo_V1.asc",
        "Exp/Beam8_110Deg_PosB/CardStella_0122_histo_V1.asc",
        "Exp/Beam9_120Deg_PosB/CardStella_0122_histo_V1.asc",
        "Exp/Beam12_125Deg_PosB/CardStella_0122_histo_V1.asc",
        "Exp/Beam10_130Deg_PosB/CardStella_0122_histo_V1.asc"
    ])

    # experiment parameters
    detector_solid_angle = 2*np.pi * (1. - 1. / np.sqrt(1. + (0.5/123.)**2)) #solid angle occupied by the detector
    Z1,A1 = 1,1
    E_beam = 3. #lab frame energy of the beam particles
    target_angle = 30 * np.pi/180
    relative_uncertainty = 2.*np.pi/180 * cotan(target_angle)
    print(f"relative uncertainty = {relative_uncertainty * 100:.2f} %")

    theta_start = 50 #lab frame angle in degrees
    theta_stop = 150 #lab frame angle in degrees
    angles = np.linspace(theta_start, theta_stop, 100)

    theoretical_au_cross_sections = []
    #theoretical_c_cross_sections = []
    for angle in angles:
        au_collision = Collision(angle * np.pi/180, detector_solid_angle, Z1, A1, 79, 197, E_beam)
        theoretical_au_cross_sections.append(au_collision.compute_lab_rutherford_cross_section())
        #c_collision = Collision(angle * np.pi/180, detector_solid_angle, Z1, A1, 6, 12, E_beam)
        #theoretical_c_cross_sections.append(c_collision.compute_lab_rutherford_cross_section())
    theoretical_au_cross_sections = np.array(theoretical_au_cross_sections)
    #theoretical_c_cross_sections = np.array(theoretical_c_cross_sections)

    experimental_au_cross_sections_A = []
    experimental_au_cross_sections_B = []

    for index in range(nb_files_A):

        au_collision = Collision(angles_A[index] * np.pi/180, detector_solid_angle, Z1, A1, 79, 197, E_beam)
        au_collision.set_nb_particles_from_faraday(faraday_counts_A[index])
        au_collision.set_target_density_from_mass_density(80.)

        interval = intervals_A[index]

        # extract experimental data from the file
        data = create_data_from_filename(file_list_A[index])

        # compute experimental cross-section
        nb_counts, fitted_function = data.compute_experimental_number_of_counts(interval[0], interval[1], 7.2e3, 50., 50., 0., 2., display=False) #retrieve experimental number of counts and number of counts per bin
        experimental_au_cross_sections_A.append(data.compute_experimental_cross_section(au_collision, nb_counts)) #compute cross-section

    for index in range(nb_files_B):

        au_collision = Collision(angles_B[index] * np.pi/180, detector_solid_angle, Z1, A1, 79, 197, E_beam)
        au_collision.set_nb_particles_from_faraday(faraday_counts_B[index])
        au_collision.set_target_density_from_mass_density(80.)

        interval = intervals_B[index]

        # extract experimental data from the file
        data = create_data_from_filename(file_list_B[index])

        # compute experimental cross-section
        nb_counts, fitted_function = data.compute_experimental_number_of_counts(interval[0], interval[1], 7.2e3, 50., 50., 0., 2., display=False) #retrieve experimental number of counts and number of counts per bin
        experimental_au_cross_sections_B.append(data.compute_experimental_cross_section(au_collision, nb_counts)) #compute cross-section

    experimental_au_cross_sections_A = np.array(experimental_au_cross_sections_A)
    au_uncertainties_A = relative_uncertainty * experimental_au_cross_sections_A
    experimental_au_cross_sections_B = np.array(experimental_au_cross_sections_B)
    au_uncertainties_B = relative_uncertainty * experimental_au_cross_sections_B

    # plot
    plt.rcParams.update({
        'font.family': 'serif',   # Type of font
        'xtick.labelsize': '20',  # Size of the xtick label
        'ytick.labelsize': '20',  # Size of the ytick label
        'lines.linewidth': '2',   # Width of the curves
        'legend.framealpha': '1', # Transparency of the legend frame
        'legend.fontsize': '23',  # Size of the legend
        'axes.labelsize' : '23',  # Size of the axes' labels
        'grid.linestyle':'--',    # Grid formed by dashed lines
        'text.usetex': True       # Using LaTex style for text and equation
    })

    plt.figure()

    plt.errorbar(angles_A, experimental_au_cross_sections_A, yerr=au_uncertainties_A, linestyle='none', marker='.', color='r', label='Position A')
    plt.errorbar(angles_B, experimental_au_cross_sections_B + au_uncertainties_B, yerr=au_uncertainties_B, linestyle='none', marker='.', color='b', label='Position B')
    plt.plot(angles, theoretical_au_cross_sections, 'k-', label='Rutherford')

    plt.xlabel('Scattering angle (°)')
    plt.ylabel('Cross-section (mb/sr)')

    plt.yscale('log')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_carbon_gold_cross_section()
    #plot_mott_section()
    #calibration()
    #theoretical_nb_counts()
    #fit_peak()
    #get_experimental_cross_sections()







