function q=cel(K,x0,u,tm,ym)
% ym-pomiary, tm-chwile próbkowania
[t,x]=rk4(x0,u,tm(end),K);
e=x-ym;
q=e'*e/length(u);   
