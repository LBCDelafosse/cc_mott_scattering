import numpy as np
import scipy.constants as cst
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from collision import *
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
    plot([angles for _ in range(2)], [au_cross_sections, c_cross_sections], ['r--', 'k--'], ['Gold', 'Carbon'], xlabel, ylabel, log=True, show=True)

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
        spin0_collision = Collision(angle, 0, Z1, A1, Z1, A1, E_beam, total_spin=0)
        spin0_collision.set_detector_angle_from_COM_angle(angle)
        spin0_cross_sections.append(spin0_collision.compute_mott_cross_section())
        spin1_collision = Collision(angle, 0, Z1, A1, Z1, A1, E_beam, total_spin=1)
        spin1_collision.set_detector_angle_from_COM_angle(angle)
        spin1_cross_sections.append(spin1_collision.compute_mott_cross_section())
        spin2_collision = Collision(angle, 0, Z1, A1, Z1, A1, E_beam, total_spin=2)
        spin2_collision.set_detector_angle_from_COM_angle(angle)
        spin2_cross_sections.append(spin2_collision.compute_mott_cross_section())

    xlabel = 'Scattering angle (°)'
    ylabel = 'Cross-section (mb/sr)'
    angles *= 180/np.pi #conversion in degrees
    plot([angles for _ in range(3)], [spin0_cross_sections, spin1_cross_sections, spin2_cross_sections], ['k-','b--','r:'], ['I=0','I=1','I=2'], xlabel, ylabel, log=True, show=True)

if __name__ == "__main__":
    #plot_carbon_gold_cross_section()
    plot_mott_section()

