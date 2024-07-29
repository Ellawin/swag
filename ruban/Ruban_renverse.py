import RCWA_3D_python.base as base
import RCWA_3D_python.materials as mat
import RCWA_3D_python.bunch as bunch
import numpy as np
import sys # DEBUGGING
import matplotlib.pyplot as plt
import PyMoosh as pm
np.set_printoptions(threshold=sys.maxsize,linewidth=110)# DEBUGGING

Mm = 5
Nm = 5
eta = 0.99

h_air = 100.0123
h_ruban = 30.12312
h_spacer = 3
h_metal = 100.23540
#pml = 40.123
l_air = 100.00232
l_ruban = h_ruban
e_spa = 1.0

theta = 0.01
phi = 0.002
#phi = 90 * np.pi / 180

nb_lamb = 1
lambdas = [700.002354]

pi = np.pi

top = bunch.Bunch()
sub = bunch.Bunch()

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

sub.eta=eta
sub.pmlx=[0,0,0]
sub.pmly=[0,0,0,0]
sub.ox=[0, l_air, l_air+l_ruban, 2*l_air+l_ruban, 2*l_air+l_ruban]
sub.nx=[0, l_air, l_air+l_ruban, 2*l_air+l_ruban, 2*l_air+l_ruban]
sub.oy=[0, h_air, h_air+h_ruban, h_air+h_ruban+h_spacer, h_air+h_ruban+h_spacer+h_metal]
sub.ny=[0, h_air, h_air+h_ruban, h_air+h_ruban+h_spacer, h_air+h_ruban+h_spacer+h_metal]
sub.Mm=Mm
sub.Nm=Nm
sub.mu=np.array([[1, 1, 1],
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
sub.k0=k0

sub.eps=np.array([[1, 1, 1],
                    [1, e_ag, 1],
                    [e_spa, e_spa, e_spa],
                    [e_ag, e_ag, e_ag]])

### avec pymoosh
# material_list = [1., 'Silver']
# layer_down = [1,0,1]
# start_index_eff = 4
# tol = 1e-12
# step_max = 1000000
# thicknesses_down = [100,0,100]
# Layer_down = pm.Structure(material_list, layer_down, thicknesses_down)
# neff_pm = pm.steepest(start_index_eff, tol, step_max, Layer_down, lambdas[0], 1)

neff_pm = 4.001749845220168-0.004834102582483683j

top.a0 = 0
top.b0 = 0

[P0, V0] = base.reseau(top)
index0 = np.argmin(abs(V0 - neff_pm * top.k0))
neff = V0[index0] / top.k0    

list_theta = np.linspace(0,90,100) / 180 * np.pi
neff_gp = np.empty(list_theta.size, dtype = complex)
r = np.empty(list_theta.size, dtype = complex)

for idx, theta in enumerate(list_theta):
    print(idx)
    #print(lambd)
    # print(mat.epsAubb(600))
   
    #a = -k0 * neff * np.sin(theta) * np.cos(phi)
    a = 0
    top.a0 = a
    b = - neff * k0 #* np.sin(theta) #* np.sin(phi)
    top.b0 = b

    sub.a0 = 0
    sub.b0 = 0

    [Pgp, Vgp] = base.reseau(top)
    #print(Vgp)
    index_gp = np.argmin(abs(Vgp - neff_pm * top.k0))
    #neff_gp[idx] = np.real(Vgp[index_gp] / top.k0) # indice effectif < à PM en réelle, partie imaginaire probablement aussi et la plus faible possible (mais attention le SP a aussi une partie imaginaire faible)
    neff_gp[idx] = Vgp[index_gp] / top.k0
    print("position mode = ", index_gp)
    print("valeur neff gp = ", neff_gp[idx])

    [Psp, Vsp] = base.reseau(sub)
    S = base.interface(Pgp, Psp)
    r[idx] = S[index_gp, index_gp]

plt.figure(3)
plt.plot(list_theta * 180 / np.pi, np.abs(r) ** 2)
plt.xlabel("Theta")
plt.ylabel("$r_{GP}$")
plt.ylim([0,1])
plt.title("Reflexion of GP")
plt.show(block=False)
plt.savefig("Rgp_vue_dessus.jpg")

plt.figure(4)
plt.title("Effective index of GP")
#plt.subplot(211)
plt.plot(list_theta * 180 / np.pi, np.real(neff_gp))
#plt.ylabel("real part")
#plt.subplot(212)
plt.figure(5)
plt.plot(list_theta * 180 / np.pi, np.imag(neff_gp))
#plt.xlabel("Theta")
#plt.ylabel("imaginary part")
plt.show(block=False)
#plt.savefig("Ngp_vue_dessus.jpg")

# plt.figure(2)
# plt.plot(list_theta * 180 / np.pi, neff_gp)
# plt.xlabel("Theta")
# plt.legend()
# plt.ylabel("$n_{GP}$")
# plt.title("Effective index")
# plt.show(block=False)
#plt.savefig("test3D_neff_gap10.jpg")
