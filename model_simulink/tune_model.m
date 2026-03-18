clear all; clc;
addpath ../Identyfikacja_dane/

% --- 1. ŁADOWANIE I PRZYGOTOWANIE OBU ZESTAWÓW DANYCH ---
files = {'square_2.mat', 'square_3.mat'};
startSample = 10;
maxTime = 25; % OGRANICZENIE CZASU DO 25 SEKUND
ds = struct(); % Struktura na dane pomocnicze

for i = 1:length(files)
    data = load(files{i});
    
    % Wyznaczamy wektory od startSample
    full_t = data.StateData.time(startSample:end) - data.StateData.time(startSample);
    full_vel = data.StateData.signals(4).values(startSample:end);
    full_ctrl = data.StateData.signals(1).values(startSample:end);
    full_pos = data.StateData.signals(3).values(startSample:end);
    
    % FILTROWANIE DO 25 SEKUND
    idx = full_t <= maxTime;
    ds(i).t = full_t(idx);
    ds(i).vel_target = full_vel(idx);
    
    % Przygotowanie sygnału sterującego dla przyciętego czasu
    ds(i).simin = timeseries(full_ctrl(idx), ds(i).t);
    
    % Warunki początkowe (z momentu startSample)
    ds(i).x0 = [full_vel(1), full_pos(1)];
end

% Stałe flagi
withMotor = 1;
useSignForFrict = 0;

% --- 2. PARAMETRY POCZĄTKOWE ---
% [Gu, Gfi, G_t, G_tp, G_fii]
Gu      = 457.443480;
Gfi     = -1.045953;
G_t     = 4.097306;
G_tp    = -0.152144;
G_fii   = -0.008456;
p0 = [Gu, Gfi, G_t, G_tp, G_fii];

% --- 3. OPTYMALIZACJA ---
options = optimset('Display', 'iter', 'TolX', 1e-4, 'MaxFunEvals', 1000);

fprintf('Rozpoczynam optymalizację na zestawach: %s i %s...\n', files{1}, files{2});

% Przekazujemy strukturę 'ds' do funkcji kosztu
[p_opt, fval] = fminsearch(@(p) cost_function_multi(p, ds, withMotor, useSignForFrict), p0, options);

% --- 4. WYŚWIETLENIE I PRINTOWANIE PARAMETRÓW ---
fprintf('\n\n');
fprintf(' OPTYMALIZACJA ZAKOŃCZONA \n');
fprintf('\n');
fprintf('Gu      = %.6f\n', p_opt(1));
fprintf('Gfi     = %.6f\n', p_opt(2));
fprintf('G_t     = %.6f\n', p_opt(3));
fprintf('G_tp    = %.6f\n', p_opt(4));
fprintf('G_fii   = %.6f\n', p_opt(5));
fprintf('\n');

% Opcjonalnie: Symulacja sprawdzająca dla obu zestawów po optymalizacji
figure('Name', 'Weryfikacja po optymalizacji');
for i = 1:2
    cost_function_multi(p_opt, ds, withMotor, useSignForFrict); % Ustaw parametry w Base
    assignin('base', 'simin', ds(i).simin);
    assignin('base', 'x0_state', ds(i).x0);
    out = sim("pendulum_model", 'StopTime', string(ds(i).t(end)));
    
    subplot(2,1,i);
    plot(ds(i).t, ds(i).vel_target, 'b'); hold on;
    plot(out.simout.Time, out.simout.Data(:, 4), 'r--');
    title(['Zestaw: ', files{i}]);
    legend('Pomiar', 'Model'); grid on;
end

%% --- FUNKCJA KOSZTU DLA WIELU ZESTAWÓW ---
function J_total = cost_function_multi(p, ds, withMotor, useSignForFrict)
    % 1. Wrzucamy parametry fizyczne do Base Workspace (wspólne dla obu symulacji)
    assignin('base', 'Gu', p(1));
    assignin('base', 'Gfi', p(2));
    assignin('base', 'G_t', p(3));
    assignin('base', 'G_tp', p(4));
    assignin('base', 'G_fii', p(5));
    assignin('base', 'withMotor', withMotor);
    assignin('base', 'useSignForFrict', useSignForFrict);
    
    J_total = 0;
    
    % 2. Iterujemy po wszystkich zestawach danych
    for i = 1:length(ds)
        % Wrzucamy specyficzne dane dla danej symulacji
        assignin('base', 'simin', ds(i).simin);
        assignin('base', 'x0_state', ds(i).x0);
        
        try
            % Uruchomienie symulacji
            out = sim("pendulum_model", 'StopTime', string(ds(i).t(end)), 'CaptureErrors', 'on');
            
            sim_t = out.simout.Time;
            sim_v = out.simout.Data(:, 4); 
            
            % Interpolacja do czasu rzeczywistego
            sim_v_interp = interp1(sim_t, sim_v, ds(i).t, 'linear', 'extrap');
            
            % Sumowanie błędu MSE dla tego zestawu
            J_total = J_total + sum((ds(i).vel_target - sim_v_interp).^2);
            
        catch
            J_total = J_total + 1e15; % Kara za błąd w którejkolwiek symulacji
        end
    end
end