load square_1.mat

t = StateData.time;
s = StateData.signals;

% for i = 1:8
%     subplot(2, 4, i)
%     plot(t, s(i).values)
%     title(s(i).label)
% 
% end

u = s(1).values;
vel = s(6).values;

plot(t, u*200, t, vel)


