clear
lambda=600
e_ag=epsAgbb(lambda);

gp.a0=0;
gp.ox=[0,70,80,300];
gp.nx=gp.ox;
gp.oy=[0,10];
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

[P1,V1]=reseau(gp);
[a,b]=min(imag(V1));
n_eff=real(V1(b)/gp.k0);

[P2, V2] = reseau(plan);
[c,d] = min(imag(V2));
n_sp = real(V2(d)/plan.k0)

disp('N_eff')
disp(n_eff)

S = interface(P1,P2);
R_GP = S(b,b)**2;
phase_R_GP = angle(S(b,b));

disp('R GP')
disp(R_GP)

disp('Phase')
disp(phase_R_GP)

disp("position GP")
disp(b)

#disp("V GP")
#disp(V1)



