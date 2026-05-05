function y = forward1(x, a)
y = a.W3 * max(a.W2 * max(a.W1 * x + a.b1(:), 0) + a.b2(:), 0) + a.b3(:);
end
