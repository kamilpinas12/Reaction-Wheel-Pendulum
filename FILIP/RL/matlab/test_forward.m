% Test forward1 using a few observations
S = load('matlab/agent_weights.mat');
agent = S.agent;
torques = S.torques;
obs_list = {S.obs_seed(:), S.obs_seed(:) + [0.05; 0; 0; 0], S.obs_seed(:) + [0; -0.05; 0.02; 0]};
for k = 1:numel(obs_list)
	logits = forward1(obs_list{k}, agent);
	[~, idx] = max(logits);
	fprintf('case %d: action=%d torque=%g\n', k, idx - 1, torques(idx));
end
