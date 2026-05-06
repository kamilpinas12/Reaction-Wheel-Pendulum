function e=cel(p,u,tm,ym,x0)
tf=tm(end);
[~,x]=rk4(x0,u,tf,p);
e=x(:,1)-ym(:,1);

