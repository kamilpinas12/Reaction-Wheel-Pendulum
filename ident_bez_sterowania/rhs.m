function dx=rhs(t,x,u,a,b)

dx=zeros(2,1);

dx(1)=x(2);
dx(2)=-a*x(2)+b*sin(x(1));
