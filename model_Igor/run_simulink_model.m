clear all; clc;
addpath ../Identyfikacja_dane/
% load square_3.mat
nazwa = 'square_2.mat';
load(nazwa)
% ZESTAW 0
% Gu = 459.2429;
% Gfi = -1.0669;
% period = 3.1631;
% G_t = (2*pi/period)^2;
% G_tp = -0.15;
% G_fii = -0.0084;

% ZESTAW 1
% Gu = 463.6562
% Gfi = -1.0745
% G_t = 4.0726
% G_tp = -0.1511
% G_fii = -0.0083

% ZESTAW 2 (z sign)
Gu      = 457.443480;
Gfi     = -1.045953;
G_t     = 4.097306;
G_tp    = -0.152144;
G_fii   = -0.008456;

% ZESTAW 3 (bez sign)
% Gu      = 454.223125;
% Gfi     = -1.182804;
% G_t     = 3.671648;
% G_tp    = -0.138275;
% G_fii   = -0.009426;

% inne nastawy?
% Gu = 51.0397;
% Gfi = -1.2703;
% G_fii =-0.084

% control params
useSignForFrict = 1;
withMotor = 1;

startSample = 10;
t = StateData.time(1:end-startSample+1);
ctrl = StateData.signals(1).values(startSample:end);
pendPosZD = StateData.signals(2).values(startSample:end);
pendPosZU = StateData.signals(3).values(startSample:end);
pendVel = StateData.signals(4).values(startSample:end);
diskPos = StateData.signals(5).values(startSample:end);
diskVel = StateData.signals(6).values(startSample:end);


simin = timeseries(ctrl , t);
x0_state = [pendVel(1), pendPosZU(1)];
simtime = t(end);
out = sim("pendulum_model", 'StopTime', string(simtime));

sim_t = out.simout.Time;
sim_data = out.simout.Data;


figure('Name', 'Porównanie Pomiarów z Symulacją', 'NumberTitle', 'off');
titles = {'Sygnał sterujący (ctrl)', 'Pozycja Wahadła Zero Down (pendPosZD)', ...
          'Pozycja Wahadła Zero Up (pendPosZU)', 'Prędkość Wahadła (pendVel)', ...
          'Pozycja Dysku (diskPos)', 'Prędkość Dysku (diskVel)'};
measured_data = {ctrl, pendPosZD, pendPosZU, pendVel, diskPos, diskVel};
for i = 1:6
    subplot(3, 2, i); 
    plot(t, measured_data{i}, 'b', 'LineWidth', 1.5); 
    hold on;
    plot(sim_t, sim_data(:, i), 'r--', 'LineWidth', 1.2); 
    title(titles{i});
    xlabel('Czas [s]');
    ylabel('Amplituda');
    grid on;
    if i == 1
        legend('Pomiary', 'Symulacja', 'Location', 'best');
    end
end
sgtitle("Porównanie wyników dla " + nazwa)