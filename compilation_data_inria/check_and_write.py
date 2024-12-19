import numpy as np
import matplotlib.pyplot as plt

# data = np.load("data_ITO_GLASS_Au10.npz")
# list_wavelengths = data['list_wavelength']
# R = data['R']
# R_ITO = R[0]
# R_AlGlass = R[1]

# plt.figure(1)
# plt.plot(list_wavelengths, R_ITO)
# plt.plot(list_wavelengths, R_AlGlass)

# plt.show()

# file = open(f"Data_structure1.txt", 'w')
# file.write("Environnement : Air / n=1 \n")
# file.write("Cube : Argent / n(lambda) / 30 nm\n")
# file.write("Gap diélectrique / n = 1.45 / 2 nm\n")
# file.write("Fonctionnalisation diélectrique / n = 1.45 / 1 nm\n")
# file.write("Couche métallique : Or / n(lambda) / 10 nm\n")
# file.write("Substrat : ITO / n(lambda) / 200 nm\n")
# file.write("\n")
# file.write("Wavelengths (nm) \t R\n")

# for i in range(len(list_wavelengths)):
#     file.write(f"{list_wavelengths[i]}, {R_ITO[i]}\n")
# file.close()

# file = open(f"Data_structure2.txt", 'w')
# file.write("Environnement : Air / n=1 \n")
# file.write("Cube : Argent / n(lambda) / 30 nm\n")
# file.write("Gap diélectrique / n = 1.45 / 2 nm\n")
# file.write("Fonctionnalisation diélectrique / n = 1.45 / 1 nm\n")
# file.write("Couche métallique : Or / n(lambda) / 10 nm\n")
# file.write("Couche d'accroche : Al / n(lambda) / 3 nm\n")
# file.write("Substrat : SiO2 / 1.5 / 200 nm\n")
# file.write("\n")
# file.write("Wavelengths (nm) \t R\n")

# for i in range(len(list_wavelengths)):
#     file.write(f"{list_wavelengths[i]}, {R_AlGlass[i]}\n")
# file.close()

data = np.load("data_accroches_all_Rdown-Rup.npz")
list_wavelength = data['list_wavelength']
R = data['R']
R_up = R[0]
R_down = R[1]

# list_metal = np.linspace(1,20,10)

# plt.figure(2)
# for idx_metal, thick_metal in enumerate(list_metal):
#     plt.plot(list_wavelength, R_up[idx_metal], label = f"Au thick : {int(thick_metal)} nm")

# plt.figure(3)
# for idx_metal, thick_metal in enumerate(list_metal):
#     plt.plot(list_wavelength, R_down[idx_metal], label = f"Au thick : {int(thick_metal)} nm")    

# plt.show()

# file = open(f"Data_structures3.txt", 'w')
# file.write("Environnement : Air / n=1 \n")
# file.write("Cube : Argent / n(lambda) / 30 nm\n")
# file.write("Gap diélectrique / n = 1.45 / 2 nm\n")
# file.write("Fonctionnalisation diélectrique / n = 1.45 / 1 nm\n")
# file.write("Couche métallique : Or / n(lambda) / 1-20 nm\n")
# file.write("Substrat : ITO / n(lambda) / 200 nm\n")
# file.write("\n")
# file.write("Wavelengths (nm) \t R1\t R3\t R5\t R7\t R9\t R11\t R13\t R15\t R17\t R20\n")

# for i in range(len(list_wavelength)):
#     file.write(f"{list_wavelength[i]}, {R_up[0]}\t, {R_up[1]}\t, {R_up[2]}\t,{R_up[3]}\t,{R_up[4]}\t,{R_up[5]}\t,{R_up[6]}\t,{R_up[7]}\t,{R_up[8]}\t, {R_up[9]}\n")
# file.close()

file = open(f"Data_structures4.txt", 'w')
file.write("Environnement : Air / n = 1 \n")
file.write("Cube : Argent / n(lambda) / 30 nm\n")
file.write("Gap diélectrique / n = 1.45 / 2 nm\n")
file.write("Fonctionnalisation diélectrique / n = 1.45 / 1 nm\n")
file.write("Couche métallique : Or / n(lambda) / 1-20 nm\n")
file.write("Substrat : ITO / n(lambda) / 200 nm\n")
file.write("\n")
file.write("Wavelengths (nm) \t R1\t R3\t R5\t R7\t R9\t R11\t R13\t R15\t R17\t R20\n")

for i in range(len(list_wavelength)):
    file.write(f"{list_wavelength[i]}, {R_down[0]}\t, {R_down[1]}\t, {R_down[2]}\t,{R_down[3]}\t,{R_down[4]}\t,{R_down[5]}\t,{R_down[6]}\t,{R_down[7]}\t,{R_down[8]}\t, {R_down[9]}\n")
file.close()
