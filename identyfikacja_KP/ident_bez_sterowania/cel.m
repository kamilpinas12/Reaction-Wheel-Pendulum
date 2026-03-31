function e=cel(p,u,tm,ym)
x0=p(1:2);
a=p(3);
b=p(4);
tf=tm(end);
[~,x]=rk4(x0,u,tf,a,b);
e=x(:,1)-ym(:,1);







