import numpy as np
import matplotlib.pyplot as plt
import PyMoosh as pm
from scipy.special import erf
from scipy.linalg import toeplitz, inv

i = complex(0,1)

material_list = [1., 'Silver']
stack = [0,1]
thicknesses = [300,300]

wavelength = 600
polarization = 1
theta = 0

Struct = pm.Structure(material_list, stack, thicknesses)

r, t, R, T = pm.coefficient_S(Struct, wavelength, theta, polarization)

print("r = ", r)
print("R = ", R)