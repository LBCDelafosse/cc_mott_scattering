import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as cst
import cross_sections as cs
import data_extraction as dt


# perform the calibration
data_file = "calibration_data.asc" #name of the file containing the calibration data
threshold = 15 #the threshold to filter the data
convert, std_dev, r2 = dt.calibrate(data_file, threshold, display=True)

# experiment parameters
A1,A2 = 1,197
detector_angle = 90 * np.pi/180 #angle of the detector in the lab frame in radians
det_solid_angle = 2*np.pi * (1. - 1. / np.sqrt(1. + (0.5/130.)**2)) #solid angle occupied by the detector
nb_particles = 10149 * 2e-9 / cst.e
au_target_density = 0.8 * cst.N_A / 197. #gold target density in SI units
c_target_density = 0.1 * cst.N_A / 12. #carbon target density in SI units
E_beam = 3. #lab frame energy in MeV
E_COM = E_beam * A2 / (A1 + A2)

au_det_energy_0, au_nb_counts_0 = cs.number_of_counts(detector_angle, det_solid_angle, nb_particles, au_target_density, 1, 1, 197, 79, E_COM, display=True)
'''
# take uncertainty into account
nb_points = 1000000
energy_samples = np.random.normal(E_beam, 0.05, nb_points) #randomly compute energies on a Gaussian obtained from calibration
nb_bins = 200 #number of bins used in the histogram
bin_content, bin_edges = np.histogram(energy_samples, bins=nb_bins)
particle_energies = np.zeros(nb_bins)
for index in range(nb_bins):
    particle_energies[index] = (bin_edges[index] + bin_edges[index+1]) / 2.

# compute number of counts
au_det_energy = np.zeros(nb_bins)
au_nb_counts = np.zeros(nb_bins)
c_det_energy = np.zeros(nb_bins)
c_nb_counts = np.zeros(nb_bins)
for index in range(nb_bins):
    au_det_energy[index], au_nb_counts[index] = cs.number_of_counts(detector_angle, det_solid_angle, bin_content[index], au_target_density, 1, 1, 197, 79, particle_energies[index])
    c_det_energy[index], c_nb_counts[index] = cs.number_of_counts(detector_angle, det_solid_angle, bin_content[index], c_target_density, 1, 1, 12, 6, particle_energies[index])

au_nb_counts *= nb_particles / nb_points
c_nb_counts *= nb_particles / nb_points

# plot
xlabel = 'Energy (MeV)'
ylabel = 'Number of counts'
cs.plot([au_det_energy, c_det_energy], [au_nb_counts, c_nb_counts], ['r-', 'g-'], ['Gold', 'Carbon'], xlabel, ylabel, show=False)
'''
