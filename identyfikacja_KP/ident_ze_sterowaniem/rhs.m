function dx=rhs(t,x,u,c)

dx=zeros(3,1);

Km = 484.73;
a = 0.1165;
b=-3.915;
d = 0.00229;

dx_3 = Km*(u-d*x(3));


dx(1)=x(2);
dx(2)=-a*x(2)+b*sin(x(1)) + c*dx_3;
dx(3) = dx_3;
