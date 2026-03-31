Gwc = tf(490.1, [1 1.144]); % PWM to reaction wheel velocity
I = 1; % Inertia of reaction wheel 
C1 = 0.368677;
C2 = 2.374985;
C3 = 0.004275;

FileName      = 'pendulum';       % File describing the model structure.
Order         = [3 1 3];           % Model orders [ny nu nx].
Parameters    = [1; 1; 1];         % Initial parameters. Np = 2.
InitialStates = [0; 0; 0];            % Initial initial states.
Ts            = 0;                 % Time-continuous system.
nlgr = idnlgrey(FileName, Order, Parameters, InitialStates, Ts, ...
                'Name', 'DC-motor');
StateData.signals{1}           
data = idata()