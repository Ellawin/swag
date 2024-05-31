clear
lambda = 600;
e_ag=epsAgbb(lambda);
period = 100
#N=100;
#list_period = linspace(100,600,100);
#j=1;

#for period = list_period
#    j

hcube = 30
lcube=30
lgap=3


    gp.a0=0;
    gp.ox=[0,lcube,lcube+lgap,lcube+lgap+period];
    gp.nx=gp.ox;
    gp.oy=[0,100];
    gp.ny=gp.oy;
    gp.Mm=10;
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
    n_eff=real(V1(b)/gp.k0)

    [P2, V2] = reseau(plan);
    [c,d] = min(imag(V2));
#[c,d] = min(abs(V2 - plan.k0));
    n_sp = real(V2(d)/plan.k0)


    S = interface(P1,P2);
    #S(b,b)
    R_GP = abs(S(b,b))**2
    phase_R_GP = angle(S(b,b));

    #b


