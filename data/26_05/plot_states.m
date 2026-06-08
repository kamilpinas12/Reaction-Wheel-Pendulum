lqr = load("LQR.mat");
rl = load("slabe_dane_stary_rl.mat");
nowy_rl = load("nowy_rl.mat");

figure(1)
subplot(3,1, 1)
hold on
grid on
plot(lqr.StateData.time-1.3, lqr.StateData.signals(6).values)
plot(nowy_rl .StateData.time, nowy_rl .StateData.signals(6).values)
plot(rl.StateData.time, rl.StateData.signals(6).values)
ylim([-450, 450])
xlim([0, 10])
title("Reaction wheel velocity")
legend("LQR", "RL")
% Plot additional signals for comparison
subplot(3,1, 2)
hold on
grid on
plot(lqr.StateData.time-1.3, lqr.StateData.signals(4).values)
plot(nowy_rl.StateData.time, nowy_rl.StateData.signals(4).values)
plot(rl.StateData.time, rl.StateData.signals(4).values)
ylim([-8, 8])
xlim([0, 10])
title("Pendulum velocity")
legend("LQR", "RL")
% Plot the third signal for comparison
subplot(3,1, 3)
hold on
grid on
plot(lqr.StateData.time-1.3, lqr.StateData.signals(7).values)
plot(nowy_rl.StateData.time, nowy_rl.StateData.signals(7).values)
plot(rl.StateData.time, rl.StateData.signals(4).values)
ylim([-10, 4])
xlim([0, 10])
title("Pendulum position")
legend("LQR", "RL")

figure(2)
subplot(4,1, 1)
hold on
grid on
plot(lqr.StateData.time-1.3, lqr.StateData.signals(6).values)
plot(nowy_rl .StateData.time, nowy_rl .StateData.signals(6).values)
ylim([-250, 350])
xlim([0, 10])
title("Reaction wheel velocity")
legend("LQR", "RL")
% Plot additional signals for comparison
subplot(4,1, 2)
hold on
grid on
plot(lqr.StateData.time-1.3, lqr.StateData.signals(4).values)
plot(nowy_rl.StateData.time, nowy_rl.StateData.signals(4).values)
ylim([-5, 5])
xlim([0, 10])
title("Pendulum velocity")
legend("LQR", "RL")
% Plot the third signal for comparison
subplot(4,1, 3)
hold on
grid on
plot(lqr.StateData.time-1.3, lqr.StateData.signals(7).values)
plot(nowy_rl.StateData.time, nowy_rl.StateData.signals(7).values)
ylim([-4, 4])
xlim([0, 10])
title("Pendulum position")
legend("LQR", "RL")
% Plot the third signal for comparison
subplot(4,1, 4)
hold on
grid on
plot(lqr.StateData.time-1.3, lqr.StateData.signals(1).values)
plot(nowy_rl.StateData.time, nowy_rl.StateData.signals(1).values)
ylim([-1.1, 1.1])
xlim([0, 10])
title("Control input")
legend("LQR", "RL")