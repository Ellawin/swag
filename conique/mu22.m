% mu22=mu*f/g

function M=mu22(s)
  
  n=2*s.Nm+1;
  m=2*s.Mm+1;

  for l=1:size(s.eps,1)
    v=[zeros(2*m+1,1)];
    for j=1:size(s.eps,2)
      v=v+s.mu(l,j).*tfd(s.ox(j),s.ox(j+1),s.nx(j),s.nx(j+1),s.eta,s.nx(max(size(s.nx))),m).'*(1+complex(0,1)*(s.pmlx(j)==1));
    endfor    
    T(:,:,l)=toep(v);
  endfor

  for j=1:m
    for k=1:m
      v=[zeros(2*n+1,1)];
      for l=1:size(s.eps,1)
	v=v+1./T(j,k,l).*tfd(s.oy(l),s.oy(l+1),s.ny(l),s.ny(l+1),s.eta,s.ny(max(size(s.ny))),n).'*(1+complex(0,1)*(s.pmly(l)==1));
      endfor
      M((j-1)*n+1:j*n,(k-1)*n+1:k*n)=inv(toep(v));
    endfor
  endfor

endfunction