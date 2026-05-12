function e = cel(p,u,tm,ym,x0)
%CEL Funkcja bledu dla lsqnonlin.
% p = [a; b; c]

a = p(1);
b = p(2);
c = p(3);

tf = tm(end);
[~,x] = rk4(x0, u, tf, a, b, c);

e = x(:,1) - ym(:,1);
