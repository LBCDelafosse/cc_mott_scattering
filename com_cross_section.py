import numpy as np
import scipy.constants as cst
import scipy.special as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define constants
Z1, Z2 = 8, 8  # Oxygen-16 nuclei
A1, A2 = 16, 16
r0 = 1.25  # fm (nuclear radius parameter)
a = 0.65  # fm (diffuseness)
V0 = 40 * cst.e  # Adjusted depth of nuclear potential (Joules)
W0 = 10 * cst.e  # Adjusted absorptive potential depth (Joules)
angles = [49.3, 60, 69.8, 80.3, 90]  # Example angles in degrees

# Conversion factors
fm_to_m = 1e-15
barn_to_m2 = 1e-28

# Define potentials
def woods_saxon_potential(r, R_N, a, V0):
    return -V0 / (1 + np.exp((r - R_N) / a))

def coulomb_potential(r, Rc, Z1, Z2):
    if r > Rc:
        return (Z1 * Z2 * cst.e**2) / (4 * np.pi * cst.epsilon_0 * r)
    else:
        return (Z1 * Z2 * cst.e**2) / (4 * np.pi * cst.epsilon_0 * Rc) * (3/2 - (r**2) / (2 * Rc**2))

def optical_potential(r, R_N, a, Rc):
    return woods_saxon_potential(r, R_N, a, V0) + coulomb_potential(r, Rc, Z1, Z2) - 1j * woods_saxon_potential(r, R_N, a, W0)

# Define Schrödinger equation
def schrodinger(r, y, l, E, R_N, Rc):
    u, dudr = y
    k2 = 2 * A1 * cst.m_u * E / cst.hbar**2
    V = optical_potential(r, R_N, a, Rc)
    return [dudr, (l * (l + 1) / r**2 + 2 * A1 * cst.m_u * V / cst.hbar**2 - k2) * u]

# Solve for phase shifts
def compute_phase_shifts(E, l_max=20):
    R_N = r0 * (A1**(1/3) + A2**(1/3)) * fm_to_m
    Rc = 1.2 * (A1**(1/3) + A2**(1/3)) * fm_to_m  # Grazing radius
    k = np.sqrt(2 * A1 * cst.m_u * E) / cst.hbar
    
    phase_shifts = []
    r_span = (0.01 * fm_to_m, 10 * R_N)
    r_eval = np.linspace(*r_span, 3000)  # Increased resolution
    
    for l in range(l_max + 1):
        sol = solve_ivp(schrodinger, r_span, [0, 1], args=(l, E, R_N, Rc), t_eval=r_eval)
        u_l = sol.y[0]
        delta_l = np.arctan2(u_l[-1], sp.spherical_jn(l, k * r_eval[-1]))
        phase_shifts.append(delta_l)
    
    return np.array(phase_shifts)

# Compute differential cross-section with angle dependence
def differential_cross_section(E, theta, l_max=20):
    k = np.sqrt(2 * A1 * cst.m_u * E) / cst.hbar
    phase_shifts = compute_phase_shifts(E, l_max)
    
    f_theta = sum((2*l+1) * np.exp(1j * phase_shifts[l]) * sp.legendre(l)(np.cos(np.radians(theta))) for l in range(l_max+1))
    return (np.abs(f_theta)**2) * (np.pi / k**2)

# Compute excitation function
E_COM_values = np.linspace(15e6, 40e6, 500) * cst.e  # Increased number of points for smoother curve

# Compute cross-section for each energy and angle
plt.figure(figsize=(12, 8))
for angle in angles:
    sigma_values = np.array([differential_cross_section(E, angle) for E in E_COM_values]) / barn_to_m2  # Convert to barns
    plt.plot(E_COM_values / cst.e / 1e6, sigma_values, label=f'Angle {angle}°')

# Plot results
plt.xlabel('Center-of-Mass Energy (MeV)', fontsize=14)
plt.ylabel('Differential Cross-Section (barns)', fontsize=14)
plt.title('$^{16}O + ^{16}O$ Scattering using Optical Model (Adjusted Parameters)', fontsize=16)
plt.legend()
plt.grid(True)
plt.show()
