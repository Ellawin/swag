function [P,V]=reseau(s)

n=2*s.Nm+1;
m=2*s.Mm+1;

v=[];
for j=-s.Mm:s.Mm 
  a=s.a0+2*pi*j/s.nx(max(size(s.nx)));
  v=[v,a.*ones(1,n)];
endfor
alpha=i*diag(v);

v=s.b0+2*pi*[-s.Nm:s.Nm]/s.ny(max(size(s.ny)));
v=repmat(v,1,m);
beta=i*diag(v);

tmp=inv(eps33(s));
Leh=[alpha*tmp*beta/(i*s.k0),i*s.k0*mu22(s)-alpha*tmp*alpha/(i*s.k0);-i*s.k0*mu11(s)+beta*tmp*beta/(i*s.k0),-beta*tmp*alpha/(i*s.k0)];

tmp=inv(mu33(s));
Lhe=[-alpha*tmp*beta/(i*s.k0),-i*s.k0*eps22(s)+alpha*tmp*alpha/(i*s.k0);i*s.k0*eps11(s)-beta*tmp*beta/(i*s.k0),beta*tmp*alpha/(i*s.k0)];

L=Leh*Lhe;

[tmp,V]=eig(L);

V=diag(V);

for j=1:max(size(V))
  V(j)=sqrt(-V(j));
  if (imag(V(j))<0)
    V(j)=-V(j);
  endif
endfor

P=[tmp;Lhe*tmp*diag(1./V)];

endfunction