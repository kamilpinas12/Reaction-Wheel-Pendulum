 = [0.0553, 0.02953, 0.02327, 0.019931];
 = [0.0199, 0.0251, 0.02508, 0.028093];

% 1. Dopasowanie wielomianu 1. stopnia (prostej)
% p(1) to współczynnik kierunkowy (a), p(2) to wyraz wolny (b)
p = polyfit(Ip, ml, 1);

% 2. Wygenerowanie punktów prostej do wykresu
Ip_fit = linspace(min(Ip), max(Ip), 100);
ml_fit = polyval(p, Ip_fit);

% 3. Wykres
scatter(Ip, ml, 'filled')
hold on
plot(Ip_fit, ml_fit, 'r', 'LineWidth', 2)
grid on
xlabel('Ip')
ylabel('ml')
legend('Dane', sprintf('ml = %.2f * Ip + %.2f', p(1), p(2)))