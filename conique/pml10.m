clear
lambda = 600;
e_ag=epsAgbb(lambda);

pos_pml = 10;

    gp.a0=0;
    gp.ox=[0,pos_pml,150,160,300];
    gp.nx=gp.ox;
    gp.oy=[0,1000];
    gp.ny=gp.oy;
    gp.Mm=70;
    gp.Nm=0;
    gp.mu=[1,1,1,1];
    gp.eps=[e_ag,e_ag,1,e_ag];
    gp.eta=0.99;
    gp.k0=2*pi/lambda;
    gp.pmlx=[1,0,0,0];
    gp.pmly=[0];
    gp.b0=0;

    plan=gp;
    plan.eps=[1,1,1,e_ag];
    plan.pmlx=[1,0,0,0];

    [P1,V1]=reseau(gp);
    [a,b]=min(imag(V1));

    [P2, V2] = reseau(plan);
    [c,d] = min(imag(V2));

    S = interface(P1,P2);
    R_GP = abs(S(b,b))**2
    phase_R_GP = angle(S(b,b))


