close all;
clear all

load square_1.mat

start=100;

tm = StateData.time(start:end);
ym = StateData.signals(7).values(start:end);
um = StateData.signals(1).values(start:end);
tm = tm - tm(1);


x0(1,1)=ym(1,1);
x0(2,1)=(ym(2,1) - ym(1,1)) / (tm(2)-tm(1));
x0(3,1)=StateData.signals(6).values(start:start);

c=-0.007;
xopt=c;

LB=-0.1';UB=0.1'; % granice xopt 
options=optimset('display','iter');

[xopt,resnorm,residual,exitflag,output,lambda,jacobian]=...
lsqnonlin('cel',xopt,LB,UB,options,um,tm,ym,x0);

c=xopt;
[t,x]=rk4(x0,um,tm(end),c);
subplot(2,1,1)
plot(t,x(:,1),tm,ym(:,1));grid

legend("symulacja", "pomiary");xlabel('t [s]');ylabel('x1');
title("Identyfikacja")

subplot(2,1,2)
plot(t,x(:,3),tm,StateData.signals(6).values(start:end));grid



fprintf("c = %f\n", c)








