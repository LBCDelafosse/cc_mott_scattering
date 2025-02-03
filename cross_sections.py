import numpy as np
import scipy.constants as cst
import matplotlib.pyplot as plt


def sec(x):
    return 1./np.cos(x)

def cosec(x):
    return 1./np.sin(x)

def yaxis(xaxis, plot_instructions, A1, Z1, A2, Z2, initial_v, total_spin):
    """
    xaxis: numpy array
        list of angles at which measurements are done, in radians
    plot_instructions: string
        determines the theoretical curve to plot (possible values: 'rutherford', 'ind_rutherford', 'mott', 'exp')
    A1: int
        mass number of the beam particle
    A2: int
        mass number of the target particle
    Z1: int
        charge number of the beam particle
    Z2: int
        charge number of the target particle
    initial_v: float
        initial velocity of the beam particle in the lab frame, in SI units
    total_spin: float
        total spin of the target/beam atom when they are indistinguishable
    """

    nb_entries = len(xaxis)
    mu = cst.m_n * A1 * A2 / (A1 + A2)
    k = mu * initial_v / cst.hbar
    n = (Z1*Z2 * cst.e*cst.e) / (4*np.pi*cst.epsilon_0 * initial_v * cst.hbar)

    yaxis = np.zeros(nb_entries)

    if 'rutherford' in plot_instructions:
        for index in range(nb_entries):
            theta = xaxis[index]
            yaxis[index] = n*n / (4.*k*k) * cosec(theta/2.)**4 #SI units
            yaxis[index] *= 10**(28) #conversion in barns

    if 'ind_rutherford' in plot_instructions:
        for index in range(nb_entries):
            theta = xaxis[index]
            yaxis[index] = n*n / (4.*k*k) * ( cosec(theta)**4 - cosec(np.pi - theta)**4 ) #SI units
            yaxis[index] *= 10**(28) #conversion in barns

    if 'mott' in plot_instructions:
        for index in range(nb_entries):
            theta = xaxis[index]
            yaxis[index] = n*n / (4.*k*k) * ( cosec(theta/2.)**4 + sec(theta/2.)**4 + 2*(-1)**(2*total_spin)/(2*total_spin+1) * np.cos(n * np.log(np.tan(theta/2.)**2)) * cosec(theta/2.)**2 * sec(theta/2.)**2 ) #SI units
            yaxis[index] *= 10**(28) #conversion in barns

    return yaxis

def plot(xaxis, yaxes, linetypes, labels, xlabel, ylabel):
    """
    xaxis: numpy array
        list of angles at which measurements are done, in radians
    yaxes: list of numpy arrays
        list of curves to plot
    linetypes: list of strings
        list of instructions indicating the type of curve to plot
    labels: list of strings
        list of the labels for each curve to plot
    xlabel: string
        label of the x axis
    ylabel: string
        label of the y axis
    """

    nb_curves = len(yaxes)
    xaxis *= 180/np.pi #conversion in degrees

    plt.rc('font',  family='serif')  # Type of font 
    plt.rc('xtick', labelsize='20')  # Size of the xtick label 
    plt.rc('ytick', labelsize='20')  # Size of the ytick label
    plt.rc('lines', linewidth='3')   # Width of the curves  
    plt.rc('legend', framealpha='1') # Transparency of the legend frame
    plt.rc('legend', fontsize='23')  # Size of the legend
    plt.rc('grid', linestyle='--')   # Grid formed by dashed lines 
    plt.rcParams.update({ "text.usetex": True }) # Using LaTex style for text and equation
    
    fig, ( fig1 ) = plt.subplots( figsize=(8, 6) )
    
    for index in range(nb_curves):
        fig1.plot(xaxis, yaxes[index], linetypes[index], label=labels[index])

    fig1.set_xlabel(xlabel, fontsize=23)
    fig1.set_ylabel(ylabel, fontsize=23)
    fig1.set_yscale('log')
    fig1.legend(loc='best')
    
    plt.tight_layout()
    plt.show()

def scattering_cross_section(E_COM, l_max, phase_shifts):
    """
    Calculate the scattering cross-section using partial wave expansion.

    E_COM: float
        Center-of-mass energy.
    l_max: int
        Maximum orbital angular momentum quantum number to include.
    phase_shifts: callable
        Function that returns the phase shift \( \delta_l \) for a given \( l \) and \( E_COM \).

    Returns:
        float: The scattering cross-section.
    """
    k = np.sqrt(2 * cst.m_n * E_COM) / cst.hbar  # Wavenumber in the COM frame
    sigma = 0
    for l in range(l_max + 1):
        delta_l = phase_shifts(l, E_COM)
        T_l = np.sin(delta_l) * np.exp(1j * delta_l)
        sigma += (2 * l + 1) * abs(T_l)**2
    sigma *= (4 * np.pi / k**2)
    return sigma

def example_phase_shifts(l, E_COM):
    """
    Example phase shift function for demonstration purposes.

    l: int
        Orbital angular momentum quantum number.
    E_COM: float
        Center-of-mass energy.

    Returns:
        float: Phase shift for the given \( l \) and \( E_COM \).
    """
    return np.arctan(E_COM / ((l + 1) * 10))  # Arbitrary example

def plot_scattering_cross_section():
    E_COM_axis = np.linspace(0.01, 10, 500)  # Center-of-mass energy in MeV
    l_max = 5  # Maximum l to consider

    cross_sections = [scattering_cross_section(E_COM, l_max, example_phase_shifts) for E_COM in E_COM_axis]


theta_start = 5 * np.pi/180 #initial angle for plot in radians
theta_stop = 175* np.pi/180 #final angle for plot in radians
xaxis = np.linspace(theta_start, theta_stop, 10000)
A1,Z1,A2,Z2 = 12,6,12,6 #collision on carbon
E_MeV = 35. #initial kinetic energy in the lab frame, in MeV
initial_v = np.sqrt(E_MeV*(10**6)*cst.e*2/(A1*cst.m_n))
total_spin = 6

c_yaxis = yaxis(xaxis, ['rutherford'], A1, Z1, A2, Z2, initial_v, total_spin)
A1,Z1,A2,Z2 = 12,6,197,79 #collision on gold
au_yaxis = yaxis(xaxis, ['rutherford'], A1, Z1, A2, Z2, initial_v, total_spin)

xlabel = 'Scattering angle (°)'
ylabel = 'Cross-section (b)'
plot(xaxis, [au_yaxis, c_yaxis], ['r--','k--'], ['Gold','Carbon'], xlabel, ylabel)

