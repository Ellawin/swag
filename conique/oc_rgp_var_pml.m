clear
lambda = 600;
e_ag=epsAgbb(lambda);

N=100;

#pml 0
    gp.a0=0;
    gp.ox=[0,150,160,300];
    gp.nx=gp.ox;
    gp.oy=[0,1000];
    gp.ny=gp.oy;
    gp.Mm=30;
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
    #n_eff=real(V1(b)/gp.k0)

    [P2, V2] = reseau(plan);
    [c,d] = min(imag(V2));
    #[c,d] = min(abs(V2 - plan.k0));
    #n_sp = real(V2(d)/plan.k0)

    S = interface(P1,P2);
    #S(b,b)
    R_GP_0 = abs(S(b,b))**2
    phase_R_GP_0 = angle(S(b,b))

#pml 10
    gp.a0=0;
    gp.ox=[0,10,150,160,300];
    gp.nx=gp.ox;
    gp.oy=[0,1000];
    gp.ny=gp.oy;
    gp.Mm=30;
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
    #n_eff=real(V1(b)/gp.k0)

    [P2, V2] = reseau(plan);
    [c,d] = min(imag(V2));
    #[c,d] = min(abs(V2 - plan.k0));
    #n_sp = real(V2(d)/plan.k0)

    S = interface(P1,P2);
    #S(b,b)
    R_GP_10 = abs(S(b,b))**2
    phase_R_GP_10 = angle(S(b,b))

#pml 20
    gp.a0=0;
    gp.ox=[0,20,150,160,300];
    gp.nx=gp.ox;
    gp.oy=[0,1000];
    gp.ny=gp.oy;
    gp.Mm=30;
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
    #n_eff=real(V1(b)/gp.k0)

    [P2, V2] = reseau(plan);
    [c,d] = min(imag(V2));
    #[c,d] = min(abs(V2 - plan.k0));
    #n_sp = real(V2(d)/plan.k0)

    S = interface(P1,P2);
    #S(b,b)
    R_GP_20 = abs(S(b,b))**2
    phase_R_GP_20 = angle(S(b,b))

#pml 50
    gp.a0=0;
    gp.ox=[0,50,150,160,300];
    gp.nx=gp.ox;
    gp.oy=[0,1000];
    gp.ny=gp.oy;
    gp.Mm=30;
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
    #n_eff=real(V1(b)/gp.k0)

    [P2, V2] = reseau(plan);
    [c,d] = min(imag(V2));
    #[c,d] = min(abs(V2 - plan.k0));
    #n_sp = real(V2(d)/plan.k0)

    S = interface(P1,P2);
    #S(b,b)
    R_GP_50 = abs(S(b,b))**2
    phase_R_GP_50 = angle(S(b,b))

