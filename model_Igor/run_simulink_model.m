clear all; clc;
% addpath ./data/14_04/
addpath ./data/
% load square_3.mat
nazwa = 'square_1.mat';
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
% Gu      = 457.443480;
% Gfi     = -1.045953;
% G_t     = 4.097306;
% G_tp    = -0.152144;
% G_fii   = -0.008456;

% % ZESTAW 3 (bez sign)
% Gu      = 454.223125;
% Gfi     = -1.182804;
% G_t     = 3.671648;
% G_tp    = -0.138275;
% G_fii   = -0.009426;

%% NASTAWY NA SQUARE_1, SIN
% ZESTAW 4 (bez sign)
Gu      = 457.443480;
Gfi     = -1.045953;
G_t     = 4.097306;
G_tp    = -0.152144;
G_fii   = -0.008456;

% Gu      = 457.443480;
% Gfi     = -1.045953;
% G_t     = 27.311296;
% G_tp    = -0.059984;
% G_fii   = -0.010594;


% G_t     = 27.361425
% G_tp    = -0.295180
% G_fii   = -0.012496

% inne nastawy?
% Gu = 51.0397;
% Gfi = -1.2703;
% G_fii =-0.084

% control params
useSignForFrict = 1;
withMotor = 1;

x0_state = [0, -pi];
SP = [0, 0, 0];
    
%% LQR
% linear model 
A = [0,  1, 0;
     G_t, G_tp, Gfi*G_fii;
     0,  0, Gfi];

B = [0; Gu*G_fii; Gu];
C = eye(3);
D = 0;

% LQR
Q = diag([10 10 0.01]);
R = 7000;

K = lqr(A, B, Q, R)
eig(A-B*K);

%% KALMAN FILTER

Ts = -1;
sys = ss(A,[B B],C,D,Ts,'InputName',{'u' 'w'},'OutputName','y');

Q = 2.3; 
R = 1; 

[kalmf,L,~,Mx,Z] = kalman(sys,Q,R);

%% SIMULATIOM
startSample = 10;
t = StateData.time(1:end-startSample+1);
ctrl = StateData.signals(1).values(startSample:end);

% --- FILTR BUTTERWORTHA NA SYGNAŁ STERUJĄCY ---
% 1. Obliczenie częstotliwości próbkowania na podstawie czasu
Ts = mean(diff(t)); % Średni krok czasu
Fs = 1/Ts;          % Częstotliwość próbkowania (Hz)

% 2. Parametry filtru
fc = 10; % Częstotliwość odcięcia w Hz (możesz dostosować, np. 5-20 Hz)
rzad = 2; % Rząd filtru (2 to zazwyczaj bezpieczny wybór)
[b, a] = butter(rzad, fc/(Fs/2), 'low'); % 'low' - filtr dolnoprzepustowy

% 3. Zastosowanie filtru
ctrl_filtered = filtfilt(b, a, ctrl);

pendPosZD = StateData.signals(2).values(startSample:end);
pendPosZU = StateData.signals(3).values(startSample:end);
pendVel = StateData.signals(4).values(startSample:end);
diskPos = StateData.signals(5).values(startSample:end);
diskVel = StateData.signals(6).values(startSample:end);


simin = timeseries([ctrl_filtered, pendPosZD, pendPosZU, pendVel, diskPos, diskVel] , t);
x0_state = [pendVel(1), pendPosZU(1)];
simtime = t(end);
% simtime = 20;
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

%% OBLICZANIE NRMSE ORAZ R2 I WIZUALIZACJA

% Indeksy sygnałów do analizy: 
% 2 - pendPosZD, 4 - pendVel, 5 - diskPos, 6 - diskVel
idx_to_calc = [2, 4, 5, 6];

% Inicjalizacja tablic
norm_err_over_time = zeros(length(t), length(idx_to_calc));
nrmse_vals = zeros(1, length(idx_to_calc));
r2_vals = zeros(1, length(idx_to_calc));

fprintf('\n--- Metryki dopasowania modelu (NRMSE i R^2) ---\n');

% Pętla po wybranych sygnałach
for k = 1:length(idx_to_calc)
    i = idx_to_calc(k);
    
    % Interpolacja danych symulacyjnych
    sim_interp = interp1(sim_t, sim_data(:, i), t, 'linear', 'extrap');
    
    % Prawdziwy sygnał z pomiarów
    y_true = measured_data{i};
    y_pred = sim_interp;
    
    % Błąd (różnica)
    err = y_true - y_pred;
    
    % Zakres sygnału (max - min)
    y_range = max(y_true) - min(y_true);
    if y_range == 0
        y_range = 1; % Zabezpieczenie przed dzieleniem przez zero dla stałych sygnałów
    end
    
    % Znormalizowany błąd w czasie (odniesiony do zakresu sygnału)
    norm_err_over_time(:, k) = err / y_range;
    
    % 1. Obliczenie NRMSE (w procentach)
    rmse = sqrt(mean(err.^2));
    nrmse_vals(k) = (rmse / y_range) * 100;
    
    % 2. Obliczenie współczynnika determinacji R^2
    SS_res = sum(err.^2); % Suma kwadratów reszt
    SS_tot = sum((y_true - mean(y_true)).^2); % Całkowita suma kwadratów
    if SS_tot == 0
        r2_vals(k) = NaN; % Zabezpieczenie dla sygnału bez wariancji
    else
        r2_vals(k) = 1 - (SS_res / SS_tot);
    end
    
    % Wypisanie wyniku w konsoli
    fprintf('%s:\n', titles{i});
    fprintf('  NRMSE = %.2f%%\n', nrmse_vals(k));
    fprintf('  R^2   = %.4f\n', r2_vals(k));
end

%% RYSOWANIE ZNORMALIZOWANEGO BŁĘDU W CZASIE

figure('Name', 'Analiza Błędów - Znormalizowany błąd', 'NumberTitle', 'off');
for k = 1:length(idx_to_calc)
    i = idx_to_calc(k);
    
    subplot(2, 2, k);
    % Rysujemy znormalizowany błąd w procentach (żeby łatwiej się czytało)
    plot(t, norm_err_over_time(:, k) * 100, 'k', 'LineWidth', 1.2);
    
    % Dodanie wyników do tytułu konkretnego wykresu
    title(sprintf('%s\nNRMSE: %.1f%% | R^2: %.3f', titles{i}, nrmse_vals(k), r2_vals(k)), ...
          'Interpreter', 'none', 'FontSize', 10);
    
    xlabel('Czas [s]');
    ylabel('Błąd [% zakresu]');
    grid on;
end
sgtitle("Znormalizowany błąd w czasie dla " + nazwa);