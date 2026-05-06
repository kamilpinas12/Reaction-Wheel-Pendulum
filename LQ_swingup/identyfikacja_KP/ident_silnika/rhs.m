function dx=rhs(t,x,u,K)

H = x*0.0023;
dx=K*(u-H);