% Pobieramy parametry pierwszego zestawu danych
simIn = Simulink.SimulationInput('pendulum_model');
simIn = simIn.setVariable('G_t', p_nominal(1));
simIn = simIn.setVariable('G_tp', p_nominal(2));
simIn = simIn.setVariable('G_fii', p_nominal(3));
simIn = simIn.setVariable('withMotor', withMotor);
simIn = simIn.setVariable('useSignForFrict', useSignForFrict);
simIn = simIn.setVariable('simin', ds(1).simin);
simIn = simIn.setVariable('x0_state', ds(1).x0);
% Dodajemy brakujące Gu i Gfi, których model prawdopodobnie potrzebuje!
simIn = simIn.setVariable('Gu', Gu);
simIn = simIn.setVariable('Gfi', Gfi);

simIn = simIn.setModelParameter('StopTime', string(ds(1).t(end)));
simIn = simIn.setModelParameter('CaptureErrors', 'on');

% Symulacja i wyświetlenie błędu
out = sim(simIn);
if ~isempty(out.ErrorMessage)
    fprintf('\n!!! ZNALEZIONO BŁĄD SYMULACJI !!!\n\n');
    disp(out.ErrorMessage);
else
    fprintf('\nBrak błędu! Symulacja przeszła pomyślnie.\n');
end