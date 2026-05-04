function logits = forward_agent(obs, agent)
% logits = forward_agent(obs, agent)
% obs: column vector (nx1)
% agent: struct with fields W1,b1,W2,b2,...

fn = fieldnames(agent);
% collect W and b by index
Ws = {};
bs = {};
for i=1:numel(fn)
    f = fn{i};
    if startsWith(f, 'W')
        idx = sscanf(f, 'W%d');
        Ws{idx} = agent.(f);
    elseif startsWith(f, 'b')
        idx = sscanf(f, 'b%d');
        bs{idx} = agent.(f);
    end
end
n = numel(Ws);
a = double(obs);
for i=1:n
    bvec = bs{i}(:);
    a = Ws{i} * a + bvec;
    if i < n
        a = max(a, 0);
    end
end
logits = a;
end
