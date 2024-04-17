clear
lambda = 600;
e_ag=epsAgbb(lambda);

N=100;
list_period = linspace(100,600,100);
j=1;

for period = list_period
    j
    gp.a0=0;
    gp.ox=[0,80,90,period];
    gp.nx=gp.ox;
    gp.oy=[0,100];
    gp.ny=gp.oy;
    gp.Mm=70;
    gp.Nm=0;
    gp.mu=[1,1,1];
    gp.eps=[e_ag,1,e_ag];
    gp.eta=0.99;
    gp.k0=2*pi/lambda;
    gp.pmlx=[0,0,0];
    gp.pmly=[0];
    gp.b0=0;

    plan=gp;
    plan.eps=[1,1,e_ag];
    plan.pmlx=[0,0,0];


#disp('avec pml')

    [P1,V1]=reseau(gp);
    [a,b]=min(imag(V1));
    #n_eff=real(V1(b)/gp.k0)

    [P2, V2] = reseau(plan);
    [c,d] = min(imag(V2));
#[c,d] = min(abs(V2 - plan.k0));
    #n_sp = real(V2(d)/plan.k0)


    S = interface(P1,P2);
    #S(b,b)
    R_GP(j) = abs(S(b,b))**2;
    phase_R_GP(j) = angle(S(b,b));

    j= j+1;
    #b
endfor

figure(1)
plot(list_period, R_GP, 'linewidth', 3)
xlabel('Period (nm)')
ylabel('R GP')
savefig('Rgp_octave_period.jpg')

figure(2)
plot(list_period, phase_R_GP, 'linewidth', 3)
xlabel('Period (nm)')
ylabel('Phase R GP')
savefig('Phase_Rgp_octave_period.jpg')