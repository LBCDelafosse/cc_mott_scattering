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
    """
    Unfinished function
    """
    E_COM_axis = np.linspace(0.01, 10, 500)  # Center-of-mass energy in MeV
    l_max = 5  # Maximum l to consider

    cross_sections = [
        scattering_cross_section(E_COM, l_max, example_phase_shifts)
        for E_COM in E_COM_axis
    ]