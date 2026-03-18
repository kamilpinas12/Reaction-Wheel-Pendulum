Km = 484.73;
a = 0.1165;
b =-3.915;
c = -0.008; 
d = 0.00229;

x0 = [pi+0.3 ; 0; 0];
SP = [pi, 0, 0];

% linear model 
A = [0,  1, 0;
     -b, -a, -Km*d*c;
     0,  0, -Km*d ];

B = [0; Km*c; Km];
C = eye(3);
D = 0;


% LQR
Q = diag([10 10 0.01]);
R = 5000;

K = lqr(A, B, Q, R);

eig(A-B*K)