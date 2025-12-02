import numpy as np
import matplotlib.pyplot as plt

file= open("pvp0/Sensitivities_gold=30_pvp=0.txt", 'r')
list_wavelength = []
Rd = []
Rd_mol = []
Ru = []
Ru_mol = []
Td= []
Td_mol = []
Tu = []
Tu_mol = []
S_lam_reso = []

for line in file:
    all_lines = file.readlines()

interesting_lines = all_lines[10:111]
#print(interesting_lines)

for line in interesting_lines:
    line = line.rstrip().split(',')
    print("The line is", line)
    print("The line[0] is : ", line[0])
    print("The float(line[0]) is : ", float(line[0]))
    list_wavelength.append(float(line[0]))   
    Rd.append(float(line[1]))
    Rd_mol.append(float(line[2]))
    Ru.append(float(line[3]))
    Ru_mol.append(float(line[4]))
    Td.append(float(line[5]))
    Td_mol.append(float(line[6]))
    Tu.append(float(line[7]))
    Tu_mol.append(float(line[8])) 
    S_lam_reso.append(float(line[9]))

#print("list list_gold : ", list_wavelength)
#print('There are', count, 'lines in the file')

list_wavelength = np.array(list_wavelength)
#print("arrat list_gold : ", list_wavelength)
Rd = np.array(Rd)
Rd_mol = np.array(Rd_mol)
Ru = np.array(Ru)
Ru_mol = np.array(Ru_mol)
Td= np.array(Td)
Td_mol = np.array(Td_mol)
Tu = np.array(Tu)
Tu_mol = np.array(Tu_mol)
S_lam_reso = np.array(S_lam_reso)


plt.figure(1)
plt.plot(list_wavelength, Ru, label = "ref")
plt.plot(list_wavelength, Ru_mol, label = "mol")
plt.legend()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Ru')
plt.show()
