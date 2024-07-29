import RCWA_3D_python.base as base
import RCWA_3D_python.materials as mat
import RCWA_3D_python.bunch as bunch
import numpy as np
import sys # DEBUGGING
import matplotlib.pyplot as plt
import PyMoosh as pm
np.set_printoptions(threshold=sys.maxsize,linewidth=110)# DEBUGGING

Mm = 15
Nm = 15
eta = 0.99

h_air = 100.0123
h_ruban = 70.12312
h_spacer = 10
h_metal = 100.23540
#pml = 40.123
l_air = 100.00232
l_ruban = h_ruban
e_spa = 1.0

theta = 0.00
phi = 0.00
#phi = 90 * np.pi / 180

nb_lamb = 1
lambdas = [700.002354]

pi = np.pi

top = bunch.Bunch()

top.eta=eta
top.pmlx=[0,0,0]
top.pmly=[0,0,0,0]
top.ox=[0, l_air, l_air+l_ruban, 2*l_air+l_ruban, 2*l_air+l_ruban]
top.nx=[0, l_air, l_air+l_ruban, 2*l_air+l_ruban, 2*l_air+l_ruban]
top.oy=[0, h_air, h_air+h_ruban, h_air+h_ruban+h_spacer, h_air+h_ruban+h_spacer+h_metal, h_air+h_ruban+h_spacer+h_metal]
top.ny=[0, h_air, h_air+h_ruban, h_air+h_ruban+h_spacer, h_air+h_ruban+h_spacer+h_metal, h_air+h_ruban+h_spacer+h_metal]
top.Mm=Mm
top.Nm=Nm
top.mu=np.array([[1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1]])

#e_au = mat.epsAubb(lambdas[0])
e_ag = mat.epsAgbb(lambdas[0])
k0 = 2*pi/lambdas[0]
top.k0=k0

top.eps=np.array([[1, 1, 1],
                    [1, e_ag, 1],
                    [e_spa, e_spa, e_spa],
                    [e_ag, e_ag, e_ag]])

### avec pymoosh
# material_list = [1., 'Silver']
# layer_down = [1,0,1]
# start_index_eff = 4
# tol = 1e-12
# step_max = 1000000
# thicknesses_down = [100,10,100]
# Layer_down = pm.Structure(material_list, layer_down, thicknesses_down)
# neff_pm = pm.steepest(start_index_eff, tol, step_max, Layer_down, lambdas[0], 1)

neff_pm = 2.670775866534444+0.0655581428061723j

top.a0 = 0
top.b0 = 0

# [P0, V0] = base.reseau(top)
# index0 = np.argmin(abs(V0 - neff_pm * top.k0))
# neff = V0[index0] / top.k0    

#list_theta = np.linspace(0,90,100) / 180 * np.pi
#neff_gp = np.empty(list_theta.size, dtype = complex)
#r = np.empty(list_theta.size, dtype = complex)

#for idx, theta in enumerate(list_theta):
    #print(idx)
    #print(lambd)
    # print(mat.epsAubb(600))
   
    #a = -k0 * neff * np.sin(theta) * np.cos(phi)
a = 0
top.a0 = a
#b = - neff_pm * k0 #* np.sin(theta) #* np.sin(phi)
top.b0 = 0

[Pgp, Vgp] = base.reseau(top)
    #print(Vgp)
index_gp = np.argmin(abs(Vgp - neff_pm * top.k0))
    #neff_gp[idx] = np.real(Vgp[index_gp] / top.k0) # indice effectif < à PM en réelle, partie imaginaire probablement aussi et la plus faible possible (mais attention le SP a aussi une partie imaginaire faible)
neff_gp = Vgp[index_gp] / top.k0
print("position mode = ", index_gp)
print("valeur neff gp = ", neff_gp)

plt.figure(1)
plt.plot(np.real(Vgp) / top.k0, "+")
plt.plot(np.ones(Vgp.size) * np.real(neff_pm))

plt.figure(2)
plt.plot(np.imag(Vgp) / top.k0, "+")
plt.plot(np.ones(Vgp.size) * np.imag(neff_pm))
plt.show(block=False)


