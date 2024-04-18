clear
%lambda = 600;
%e_ag=epsAgbb(lambda);

N=100;
list_lambda = linspace(300,900,100);
j=1;

parfor lambda = list_lambda
    j
    e_ag(j) = epsAgbb(lambda);
    j= j+1;
endparfor

figure(5)
plot(list_lambda, e_ag, 'linewidth', 3)
xlabel('Wavelength (nm)')
ylabel('silver_permittivity')
savefig('silver_perm_octave.jpg')
