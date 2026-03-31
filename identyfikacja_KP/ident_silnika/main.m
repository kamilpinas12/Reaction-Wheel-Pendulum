clear;close all
load square_1.mat


start = 500;
tm=StateData.time(start:end);
ym=StateData.signals(6).values(start:end);
um=StateData.signals(1).values(start:end);
tm = tm - tm(1);
x0 = ym(1);

K = 500;
LB=-1000; UB=1000;
options=optimset('display','iter');

[xopt,resnorm,residual,exitflag,output,lambda,jacobian]=...
lsqnonlin('cel',K,LB,UB,options,x0,um,tm,ym);

fprintf("K: %f\n", xopt);


[t,x]=rk4(x0,um,tm(end),xopt);
figure(2)
plot(tm,ym,t,x, tm, um*100)
grid on
legend("Pomiary", "Model", "u*100")
xlabel("Time")
ylabel("x3")
