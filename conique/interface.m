function S=interface(P,Q)
n=size(P)(2);
U=[[P(1:n,1:n),-Q(1:n,1:n)];[P(n+1:2*n,1:n),Q(n+1:2*n,1:n)]];
V=[[-P(1:n,1:n),Q(1:n,1:n)];[P(n+1:2*n,1:n),Q(n+1:2*n,1:n)]];
S=inv(U)*V;
endfunction