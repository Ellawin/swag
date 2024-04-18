clear
%lambda = 600;
%e_ag=epsAgbb(lambda);

N=100;
list_lambda = linspace(300,900,100);
j=1;

parfor lambda = list_lambda
    j
    e_ag = epsAgbb(lambda);
    gp.a0=0;
    gp.ox=[0,80,90,300];
    gp.nx=gp.ox;
    gp.oy=[0,100];
    gp.ny=gp.oy;
    gp.Mm=30;
    gp.Nm=0;
    gp.mu=[1,1,1];
    gp.eps=[e_ag,1,e_ag];
    gp.eta=0.01;
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
    n_eff(j)=real(V1(b)/gp.k0);

    [P2, V2] = reseau(plan);
    [c,d] = min(imag(V2));
#[c,d] = min(abs(V2 - plan.k0));
    #n_sp = real(V2(d)/plan.k0)


    S = interface(P1,P2);
    #S(b,b)
    R_GP(j) = abs(S(b,b))**2;
    phase_R_GP(j) = angle(S(b,b));

    j = j+1;
    #b
endparfor

figure(7)
subplot(2,1,1)
plot(list_lambda, real(n_eff))
ylabel("Real part")
subplot(2,1,2)
plot(list_lambda, imag(n_eff))
ylabel("Imaginary part")
xlabel("Wavelength (nm)")
savefig('neff_GP_octave.jpg')

figure(6)
subplot(2,1,1)
plot(list_lambda, R_GP, 'linewidth', 2)
#xlabel('Wavelength (nm)')
ylabel('Module')
#savefig('Rgp_octave_lambda.jpg')

#figure(6)
subplot(2,1,2)
plot(list_lambda, phase_R_GP, 'linewidth', 2)
xlabel('Wavelength (nm)')
ylabel('Phase')
savefig('Rgp_octave_lambda.jpg')