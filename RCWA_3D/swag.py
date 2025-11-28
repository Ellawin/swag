import RCWA_3D_python.base as base
import RCWA_3D_python.materials as mat
import RCWA_3D_python.bunch as bunch
import numpy as np
import sys # DEBUGGING
np.set_printoptions(threshold=sys.maxsize,linewidth=110)# DEBUGGING

# def Au(lambd):

#     a = np.array([397.3853996161624,413.280815600809,430.500849584176,450.8517988372461,471.4229835750673,495.9369787209707,520.9422045388347,548.6028525674456,582.0856557757872,616.8370382101626,659.4906631927803,704.455935683197,756.0014919526993,821.08771311419,891.9729833110985,984.001941906688,1087.581093686339,1215.531810590614,1393.081400901603,1610.184995847308,1937.253823128792])
#     re = np.array([-1.649404,-1.702164,-1.692204,-1.758996,-1.702701,-2.278289,-3.946161,-5.842125,-8.112669,-10.661884,-13.648209, -16.817709,-20.610164,-25.811289,-32.040669,-40.2741,-51.04960000000001,-66.21852499999999,-90.426461,-125.3505,-189.042])
#     im = np.array([5.73888,5.717359999999999,5.6492,5.28264,4.84438,3.81264,2.58044,2.1113,1.66054,1.37424,1.03516,1.06678,1.27176, 1.62656,1.92542,2.794,3.861000000000001,5.701499999999999,8.18634,12.5552,25.3552])
#     z = re + 1.j*im
#     j = 0
#     while (lambd>a[j]):
#         j=j+1
#     j = j-1
#     Au = (z[j+1]-z[j]) / (a[j+1]-a[j])*(lambd-a[j]) + z[j]
#     print(Au)

#     return Au
# eps = np.array([-3.3497077722742450+3.0210911687875961j,-4.1557540867392362+2.5285780355745944j,-4.9782789147922468+2.3250513018723225j,-5.8008037428452575+2.1215245681700505j,-6.6149920060526650+1.9578664723720673j,-7.4287393127451917+1.7963171990865785j,-8.2530988692001177+1.6447684388614983j,-9.1333678784683041+1.5459062649473094j,-10.013636887736492+1.4470440910331206j,-10.883334085718694+1.3490956813674380j,-11.723494595250960+1.2537002467024159j,-12.563655104783226+1.1583048120373938j,-13.403815614315491+1.0629093773723719j,-14.248012653163805+1.0411437968407076j,-15.093865538487901+1.0495823161515363j,-15.939718423811998+1.0580208354623650j,-16.785571309136092+1.0664593547731940j,-17.667062190371624+1.1126870170254091j,-18.549960593255147+1.1604071604171737j,-19.432858996138673+1.2081273038089386j,-20.315757399022196+1.2558474472007035j,-21.249337730005379+1.3153618302758947j,-22.208272947795027+1.3807765731707353j,-23.167208165584675+1.4461913160655762j,-24.126143383374323+1.5116060589604168j])

#%%
Mm = 10
Nm = 0
eta = 0.999 # stretching


hcube = 30.0
hspacer = 3.0
hlayer = 10.0
l_cubex = 30.0
l_cubey = 30.0
space_x = 101-l_cubex
space_y = 102-l_cubey
eps_env = 1.0 **2
eps_dielec = 1.41 **2
eps_glass = 1.5 **2
# fin_pml = 500.01
# deb_cube = 100.01

# nb_lamb = 75
lambdas = np.linspace(400,1800,3)
#lambdas = np.concatenate((np.arange(7000,9000,400),np.arange(9000,11000,50), np.arange(11000,13000,400)))
r = np.zeros(len(lambdas), dtype=complex)
theta = 0.0 * np.pi/180. #latitude (z)
phi = 0.0 * np.pi/180. # précession (xy)
pol = 90*np.pi/180. # 0 (TE?) ou 90 (TM?) pour fixer la pola

pi = np.pi

top = bunch.Bunch()

top.ox = [0,l_cubex,l_cubex+space_x]
top.nx = [0,l_cubex,l_cubex+space_x]
top.oy = [0,l_cubey,l_cubey+space_y]
top.ny = [0,l_cubey,l_cubey+space_y]
top.Mm=Mm
top.Nm=Nm
top.mu =  np.array([[1.,1.],
                  [1.,1.]])
top.eps =  np.array([[1.,1.],
                  [1.,1.]])

top.eta=eta
top.pmlx=[0, 0]
top.pmly=[0, 0]

bot = bunch.Bunch()

bot.ox = [0,l_cubex,l_cubex+space_x]
bot.nx = [0,l_cubex,l_cubex+space_x]
bot.oy = [0,l_cubey,l_cubey+space_y]
bot.ny = [0,l_cubey,l_cubey+space_y]
bot.Mm=Mm
bot.Nm=Nm
bot.mu = np.array([[1.,1],
                  [1.,1.]])

bot.eps = np.array([[eps_glass, eps_glass], 
                   [eps_glass, eps_glass]])

bot.eta=eta
bot.pmlx=[0, 0]
bot.pmly=[0, 0]

spa = bunch.Bunch()

spa.ox = [0,l_cubex,l_cubex+space_x]
spa.nx = [0,l_cubex,l_cubex+space_x]
spa.oy = [0,l_cubey,l_cubey+space_y]
spa.ny = [0,l_cubey,l_cubey+space_y]
spa.Mm=Mm
spa.Nm=Nm
spa.mu =  np.array([[1.,1],
                  [1.,1.]])
spa.eps =  np.array([[eps_dielec,eps_dielec],
                  [eps_dielec,eps_dielec]])
spa.eta=eta
spa.pmlx=[0, 0]
spa.pmly=[0, 0]

gp = bunch.Bunch()

gp.ox = [0,l_cubex,l_cubex+space_x]
gp.nx = [0,l_cubex,l_cubex+space_x]
gp.oy = [0,l_cubey,l_cubey+space_y]
gp.ny = [0,l_cubey,l_cubey+space_y]
gp.Mm = Mm
gp.Nm = Nm
gp.mu = np.array([[1.,1],
                  [1.,1.]])
gp.eta = eta
gp.pmlx=[0, 0]
gp.pmly=[0, 0]

ml = bunch.Bunch()

ml.ox = [0,l_cubex,l_cubex+space_x]
ml.nx = [0,l_cubex,l_cubex+space_x]
ml.oy = [0,l_cubey,l_cubey+space_y]
ml.ny = [0,l_cubey,l_cubey+space_y]
ml.Mm=Mm
ml.Nm=Nm
ml.mu =  np.array([[1.,1],
                  [1.,1.]])

ml.eta=eta
ml.pmlx=[0, 0]
ml.pmly=[0, 0]

for i, lambd in enumerate(lambdas):
    # print(lambd)
    e_au = mat.epsAubb(lambd)
    e_ag = mat.epsAgbb(lambd)
    # if i == 0:
    #     e_au = Au(lambd)
    # else:
    #     e_au = eps[i-1]
    # if lambd == 800:
    #     e_au = -24.126143383374323 + 1.5116060589604168j
    # else:
    #     e_au = -64.377182754822641 + 5.4780829238651041j
    # print(lambd)
    k0 = 2*pi/lambd
    top.k0 = k0
    spa.k0 = k0
    bot.k0 = k0
    gp.k0 = k0
    ml.k0 = k0

    a = -k0 * np.sin(theta) * np.cos(phi)
    top.a0 = a
    bot.a0 = a
    spa.a0 = a
    gp.a0 = a
    ml.a0 = a 

    b = -k0 * np.sin(theta) * np.sin(phi)
    top.b0 = b
    bot.b0 = b
    spa.b0 = b
    gp.b0 = b
    ml.b0 = b 

    gp.eps =  np.array([[e_ag,1.],
                         [1.,1.]])
    ml.eps = np.array([[e_au,e_au],
                       [e_au,e_au]])
    
    [Pair,Vair], ext = base.homogene(top, ext=1)
    # print("Pair", Pair)
    # print("Vair", Vair)
    # isort = np.argsort(np.imag(Vair))
    #Vair_sort = Vair#[isort]
    #Vair_sort = np.real(Vair_sort) * (np.abs(np.real(Vair_sort))>1e-10) + 1.0j*(np.imag(Vair_sort) * (np.abs(np.imag(Vair_sort))>1e-10))
    # print("Vair sorted")
    # for i in range(len(Vair_sort)):
    #     print(Vair_sort[i])
    # print(lambd, Pair, Vair)
    [Pgp,Vgp] = base.reseau(gp)
    #print("Pgp")
    [Psub,Vsub], ext2 = base.homogene(bot, ext=1)
    #print("Psub")
    [Pspa,Vspa] = base.homogene(spa)
    #print("Pspa")
    [Pml, Vml] = base.homogene(ml)
    #print("Pml")    

#     # print("interface",base.interface(Pair, Pgp))

    S = base.c_bas(base.interface(Pair, Pgp), Vgp, hcube)
    #print("top layer", end=" ")
    # print(np.shape(Pair), np.shape(Vair))
    S = base.cascade(S, base.c_bas(base.interface(Pgp, Pspa), Vspa, hspacer))
    #print("middle layer", end=" ")
    S = base.cascade(S, base.c_bas(base.interface(Pspa, Pml), Vml, hlayer))
    S = base.cascade(S, base.c_bas(base.interface(Pml, Psub), Vsub, 0))
    #S = base.c_haut(base.interface(Pml, Psub), Vsub, hlayer) # test Pauline
    #print("bottom layer", end=" ")
    # print(S)

    # Creating the entry vector
    # print(ext)
    a = np.cos(pol) * np.cos(theta) * np.cos(phi) - np.sin(pol) * np.sin(phi)
    b = np.cos(pol) * np.cos(theta) * np.sin(phi) + np.sin(pol) * np.cos(phi)
    c = top.eps[0,0] * top.mu[0,0] * top.k0**2
    d = np.sqrt(c - top.a0**2 - top.b0**2)
    e = ((c-top.b0**2)*np.abs(a)**2 + (c-top.a0**2)*np.abs(b)**2 + 2*top.a0*top.b0*np.real(a*b)) / (top.mu[0,0]*d)
    
    V = np.zeros(4 * (2*Nm+1) *(2*Mm+1))
    V[int(np.real(ext[3,0]))] = a/np.sqrt(e)
    # ! Petite bizarrerie, ext[0,0] contient la moitié du nombre de 
    # ! modes propagatifs en espace libre. Normalement. A cause de la polarisation :
    # ! pour chaque mode, il y a deux polarisations !
    V[int(np.real(ext[3,int(np.real(ext[0,0]))]))] = b/np.sqrt(e)

    V = S @ V

    reflechi = base.efficace(top, ext, V[:2 * (2*Nm+1) *(2*Mm+1)])
    # print(reflechi)
    r[i] = reflechi[3,0]
    print(lambd, r[i])


    # print(S)
    b = np.argmin(np.imag(Vair))
    # print(b, Vair[b])
    r[i] = S[b,b]
    # print(lambd, S[b,b], abs(S[b,b]))
#     # print(Vair[b])
#     # print(np.real(Vair[b])/gp.k0)

import matplotlib.pyplot as plt
plt.figure(2)
plt.plot(lambdas, np.abs(r))
plt.xlabel("Wavelength")
plt.ylabel("|r|")
plt.ylim([-0.1,1.1])
plt.savefig("Cube30gold10gap3_lam200_Nm0.png")
plt.show(block=False)

np.savez("Cube30gold10gap3_lam200_Nm0.npz", lambdas=lambdas, r = r)
