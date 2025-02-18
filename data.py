import numpy as np
import scipy.constants as cst
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

class Data:
    """
    Object containing the parameters extracted from
    """

    # Constructors

    def __init__(self, data_file):
        self.data_file = data_file