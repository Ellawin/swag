function fft=tfd(a,b,a1,b1,eta,d,N)

fft=0.;
for n=-N:N   
  if (n==0)
    fft(N+1) = (b-a)/d;
  elseif (d-n*(b1-a1)==0)
    fft(n+N+1)=-1./(2*i*pi)*(b-a)/(b1-a1)*(1/n-eta/2*(b1-a1)/(d+n*(b1-a1)))*(exp(-2*i*pi*b1*n/d)-exp(-2*i*pi*a1*n/d))-eta/2*(b-a)*exp(-2*i*pi*a1/(b1-a1))/d;
  elseif (d+n*(b1-a1)==0)
    fft(n+N+1)=-1./(2*i*pi)*(b-a)/(b1-a1)*(1/n+eta/2*(b1-a1)/(d-n*(b1-a1)))*(exp(-2*i*pi*b1*n/d)-exp(-2*i*pi*a1*n/d))-eta/2*(b-a)*exp(2*i*pi*a1/(b1-a1))/d;
  else
    fft(n+N+1)=-1./(2*i*pi)*(b-a)/(b1-a1)*(1/n+eta/2*((b1-a1)/(d-n*(b1-a1))-(b1-a1)/(d+n*(b1-a1))))*(exp(-2*i*pi*b1*n/d)-exp(-2*i*pi*a1*n/d));
  endif
endfor



end