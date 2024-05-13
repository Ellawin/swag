clear
lambda = 600;
e_ag=epsAgbb(lambda);

N=100;
list_thick_pml = [50:250];
j=1

parfor thick_pml = list_thick_pml
    gp.a0=0;
    gp.ox=[0,thick_pml,300+thick_pml,310+thick_pml,320+thick_pml, 550+thick_pml,750+thick_pml];
    gp.nx=gp.ox;
    gp.oy=[0,10];
    gp.ny=gp.oy;
    gp.Mm=100;
    gp.Nm=0;
    gp.mu=[1,1,1,1,1,1];
    gp.eps=[1.54,1.54,e_ag, 1, e_ag, e_ag];
    gp.eta=0.99;
    gp.k0=2*pi/lambda;
    gp.pmlx=[1,0,0,0,0,1];
    gp.pmly=[0];
    gp.b0=0;

    plan=gp;
    plan.eps=[1.54,1.54,e_ag,1,1,1];
    plan.pmlx=[1,0,0,0,0,1];


#disp('avec pml')

    [P1,V1]=reseau(gp);
    neff_gp_PM = 4.02 + 0.19i;
    [a2, b2] = min(abs(V1 - neff_gp_PM * gp.k0));
    #b2
    [a,b]=min(imag(V1));
    #b
    #n_eff=real(V1(b)/gp.k0)

    [P2, V2] = reseau(plan);
    #[c,d] = min(imag(V2));
#[c,d] = min(abs(V2 - plan.k0));
    #n_sp = real(V2(d)/plan.k0)


    S = interface(P1,P2);
    #S(b,b)
    R_GP(j) = abs(S(b,b))**2;
    #phase_R_GP(j) = angle(S(b,b));

    R_GP2(j) = abs(S(b2,b2))**2;
    #phase_R_GP2(j) = angle(S(b2,b2));

    j= j+1
    #b
endparfor

figure(3)
plot(list_thick_pml, R_GP, 'linewidth', 3)
hold on
plot(list_thick_pml, R_GP2, 'linewidth', 3, "r")
xlabel('Thickness of the PML')
ylabel('R GP')
#legend('min de la partie imaginaire', 'plus proche de PM')
print('Rgp_quartz_comp.jpeg')
print('Rgp_quartz_comp.pdf')
print('Rgp_quartz_comp.eps')


### mettre que 100 points
### trouver le régime de fonctionnement qui font que PML convaincantes