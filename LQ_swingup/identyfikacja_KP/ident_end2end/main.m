close all;
clear all

% End-to-end identyfikacja na danych ze sterowaniem:
% optymalizuje parametry wahadla (a,b) oraz sprzezenie z silnikiem (c).
% Parametry silnika (Km, d) pozostaja stale (nie sa identyfikowane).

% ===========================
% WYBOR DANYCH (plik + start)
% ===========================
% Zasada: wystarczy ODKOMENTOWAC jeden z presetow ponizej.
% Nie musisz komentowac domyslnego wyboru — odkomentowany preset nizej
% nadpisze (dataFile,start). Jesli odkomentujesz kilka, wygra ostatni.

% Domyslnie (NOWE PLIKI W TYM FOLDERZE):
%dataFile = 'ident_square_53.mat';
%start = 100;
%a = 0.054164
%b = -1.931606
%c = -0.008024

%dataFile = 'ident_square_85.mat';
%start = 200;
%a = 0.299692
%b = -4.015983
%c = -0.007858

%dataFile = 'ident_square_135.mat';
%start = 400;
%a = 0.078027
%b = -8.383954
%c = -0.009725

%dataFile = 'ident_square_106.mat';
%start = 100;
%a = 0.104845
%b = -5.738891
%c = -0.008955


% --- (Opcjonalnie) pliki z ../Identyfikacja_dane/ ---
% dataFile = '../Identyfikacja_dane/square_1.mat';
% start = 100;
%
% dataFile = '../Identyfikacja_dane/square_2.mat';
% start = 100;
%
% dataFile = '../Identyfikacja_dane/square_3.mat';
% start = 100;
%
% dataFile = '../Identyfikacja_dane/sin_1.mat';
% start = 100;
%
% dataFile = '../Identyfikacja_dane/sawtooth_1.mat';
% start = 100;
%
% Uwaga: drgania_wlasne_* to eksperymenty bez sterowania; end2end nadal
% zadziala, ale identyfikacja 'c' moze byc slabo pobudzona.
% dataFile = '../Identyfikacja_dane/drgania_wlasne_1.mat';
% start = 1583;
%
% dataFile = '../Identyfikacja_dane/drgania_wlasne_2.mat';
% start = 900;

if exist(dataFile,'file') ~= 2
    error('Nie znaleziono pliku danych: %s', dataFile)
end

load(dataFile)

tm = StateData.time(start:end);
ym = StateData.signals(7).values(start:end);
um = StateData.signals(1).values(start:end);
tm = tm - tm(1);

x0(1,1) = ym(1,1);
x0(2,1) = (ym(2,1) - ym(1,1)) / (tm(2) - tm(1));
x0(3,1) = StateData.signals(6).values(start:start);

% Parametry poczatkowe (zgodne z poprzednimi identyfikacjami)
a0 = 0.165;
b0 = -1.915;
c0 = -0.008;
p0 = [a0; b0; c0];

LB = [-100; -10; -0.1];
UB = [ 100;  10;  0.1];
options = optimset('display','iter');

[popt,resnorm,residual,exitflag,output,lambda,jacobian] = ...
    lsqnonlin('cel', p0, LB, UB, options, um, tm, ym, x0);

a = popt(1);
b = popt(2);
c = popt(3);

[t,x] = rk4(x0, um, tm(end), a, b, c);

subplot(3,1,1)
plot(t, x(:,1), tm, ym(:,1)); grid
legend('symulacja', 'pomiary'); xlabel('t [s]'); ylabel('x1');
title('Identyfikacja end2end')

subplot(3,1,2)
plot(t, x(:,3), tm, StateData.signals(6).values(start:end)); grid
title('Silnik (nieidentyfikowany)')

subplot(3,1,3)
plot(t, x(:,2), tm, StateData.signals(4).values(start:end)); grid
title('Predkosc wahadla')

fprintf('a = %f\n', a)
fprintf('b = %f\n', b)
fprintf('c = %f\n', c)
