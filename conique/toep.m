function T=toep(v)
  
  n=(max(size(v))-1)/2;
  a=v(n+1:-1:2);
  b=v(n+1:2*n);
  T=toeplitz(b,a);

  end