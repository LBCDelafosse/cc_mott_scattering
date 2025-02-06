import numpy as np
import scipy.constants as cst
import matplotlib.pyplot as plt
import cross_sections as cs


detector_angle = 40 * np.pi/180 #angle of the detector in the lab frame in radians
det_solid_angle = 2*np.pi * (1. - 1. / np.sqrt(1. + (0.5/130.)**2))
duration = 1000. #acquisition time in seconds
particle_nb = duration * 30. * 6.25e9 #number of incident particles
au_target_density = 0.8 * 6.022e23 / 197. #gold target density in SI units
c_target_density = 0.1 * 6.022e23 / 12. #carbon target density in SI units
E_COM = 4. #center-of-mass energy in MeV

au_det_energy, au_nb_counts = cs.number_of_counts(detector_angle, det_solid_angle, particle_nb, au_target_density, 12, 6, 197, 79, E_COM)
c_det_energy, c_nb_counts = cs.number_of_counts(detector_angle, det_solid_angle, particle_nb, c_target_density, 12, 6, 12, 6, E_COM)

xlabel = 'Energy (MeV)'
ylabel = 'Number of counts'
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
fig1.scatter(au_det_energy, au_nb_counts, marker='o', color='red', label='Gold')
fig1.scatter(c_det_energy, c_nb_counts, marker='o', color='green', label='Carbon')
fig1.set_xlabel(xlabel, fontsize=23)
fig1.set_ylabel(ylabel, fontsize=23)
fig1.set_yscale('log')
fig1.legend(loc='best')
plt.tight_layout()
#plt.show()
#plt.savefig('ideal_nb_counts.pdf')