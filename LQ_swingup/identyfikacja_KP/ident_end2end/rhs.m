function dx = rhs(t,x,u,a,b,c)

dx = zeros(3,1);

% Parametry silnika (nieidentyfikowane w end2end)
Km = 484.73;
d  = 0.00229;

% Model: x(1) - kat, x(2) - predkosc kata, x(3) - stan silnika
% dx3 opisuje dynamike silnika (uproszczona)
dx_3 = Km*(u - d*x(3));

dx(1) = x(2);
dx(2) = -a*x(2) + b*sin(x(1)) + c*dx_3;
dx(3) = dx_3;
