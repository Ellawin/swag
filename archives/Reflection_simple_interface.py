import PyMoosh as pm 
import matplotlib.pyplot as plt
import numpy as np

material_list = [1.5**2, 1]
stack = [0,1]
thicknesses = [200,200]

situation1 = pm.Structure(material_list, stack, thicknesses)

wavelength = 600

min_theta = 0
max_theta = 90
number_theta = 100

polarization = 1

incidence, r, t, R, T = pm.angular(situation1, wavelength, polarization, min_theta, max_theta, number_theta)

plt.figure(1)

plt.subplot(2,1,1)
plt.plot(incidence, R)
plt.xlabel("Angle (in degree)")
plt.ylabel("Modulus Reflectance")
plt.title("Modulus")

plt.subplot(2,1,2)
plt.plot(incidence, np.angle(r))
plt.xlabel("Angle (in degree)")
plt.ylabel("Phase reflectance")
plt.title('Phase')

plt.subplots_adjust(hspace = 0.8)
plt.show(block=False)

