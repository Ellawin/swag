import PyMoosh as pm 
import matplotlib.pyplot as plt
import numpy as np

material_list = [1.41**2, 'Gold', 'Silver']

wavelength = 400
polarization = 1
start_index_eff = 4
tol = 1e-12
step_max = 10000

thickness_gap = 5

thicknesses = [50,thickness_gap,50]

stack_mixte = [2,0,1]
stack_gold = [1,0,1]
stack_silver = [2,0,2]

struc_mixte = pm.Structure(material_list, stack_mixte, thicknesses)
GP_effective_index_mixte = pm.steepest(start_index_eff, tol, step_max, stack_mixte, wavelength, polarization)

struc_gold = pm.Structure(material_list, stack_gold, thicknesses)
GP_effective_index_gold = pm.steepest(start_index_eff, tol, step_max, stack_gold, wavelength, polarization)

struc_silver = pm.Structure(material_list, stack_silver, thicknesses)
GP_effective_index_silver = pm.steepest(start_index_eff, tol, step_max, stack_silver, wavelength, polarization)

print("Mixte = %f" %GP_effective_index_mixte)
print("Gold = %f" %GP_effective_index_gold)
print("Silver = %f" %GP_effective_index_silver)

x_mixte,prof_mixte = pm.profile(struc_mixte, GP_effective_index_mixte,wavelength,polarization,pixel_size = 0.1)
x_gold,prof_gold = pm.profile(struc_gold, GP_effective_index_gold,wavelength,polarization,pixel_size = 0.1)
x_silver,prof_silver = pm.profile(struc_silver, GP_effective_index_silver,wavelength,polarization,pixel_size = 0.1)

plt.plot(x_mixte,np.real(prof_mixte), "r", label = "Silver and Gold")
plt.plot(x_gold,np.real(prof_gold), "b", label = "Gold")
plt.plot(x_silver,np.real(prof_silver),"g", label = "Silver")

plt.show(block=False)
plt.title("Silver and Gold")
plt.savefig("Profil_GP_lam400_silver_gold_comp.jpg")

