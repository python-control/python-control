% pvtol_lqr.m - LQR design for vectored thrust aircraft
% RMM, 14 Jan 03

aminit;

%%
%% System dynamics
%%
%% These are the dynamics for the PVTOL system, written in state space
%% form.
%%

pvtol_params;			% System parameters

% System matrices (entire plant: 2 input, 2 output)
xe = [0 0 0 0 0 0]; ue = [0 m*g];
[A, B, C, D] = pvtol_linearize(xe, ue);

%%
%% Construct inputs and outputs corresponding to steps in xy position
%%
%% The vectors xd and yd correspond to the states that are the desired
%% equilibrium states for the system.  The matrices Cx and Cy are the 
%% corresponding outputs.
%%
%% The way these vectors are used is to compute the closed loop system
%% dynamics as
%%
%%	xdot = Ax + B u		=>	xdot = (A-BK)x + K xd
%%         u = -K(x - xd)		   y = Cx
%%
%% The closed loop dynamics can be simulated using the "step" command, 
%% with K*xd as the input vector (assumes that the "input" is unit size,
%% so that xd corresponds to the desired steady state.
%%

xd = [1; 0; 0; 0; 0; 0];  Cx = [1 0 0 0 0 0];
yd = [0; 1; 0; 0; 0; 0];  Cy = [0 1 0 0 0 0];

%%
%% LQR design
%%

% Start with a diagonal weighting
Qx1 = diag([1, 1, 1, 1, 1, 1]);
Qu1a = diag([1, 1]);
K1a = lqr(A, B, Qx1, Qu1a);

% Close the loop: xdot = Ax + B K (x-xd)
H1a = ss(A-B*K1a, B*K1a*[xd, yd], [Cx; Cy], 0);
[Y, T] = step(H1a, 10);

figure(1); clf; subplot(321);
  plot(T, Y(:,1, 1), '-', T, Y(:,2, 2), '--', ...
    'Linewidth', AM_data_linewidth); hold on;
  plot([0 10], [1 1], 'k-', 'Linewidth', AM_ref_linewidth); hold on;

  amaxis([0, 10, -0.1, 1.4]); 
  xlabel('time');   ylabel('position');

  lgh = legend('x', 'y', 'Location', 'southeast');
  legend(lgh, 'boxoff');
  amprint('pvtol-lqrstep1.eps');

% Look at different input weightings
Qu1a = diag([1, 1]); K1a = lqr(A, B, Qx1, Qu1a);
H1ax = ss(A-B*K1a,B(:,1)*K1a(1,:)*xd,Cx,0);

Qu1b = 40^2*diag([1, 1]); K1b = lqr(A, B, Qx1, Qu1b);
H1bx = ss(A-B*K1b,B(:,1)*K1b(1,:)*xd,Cx,0);

Qu1c = 200^2*diag([1, 1]); K1c = lqr(A, B, Qx1, Qu1c);
H1cx = ss(A-B*K1c,B(:,1)*K1c(1,:)*xd,Cx,0);

[Y1, T1] = step(H1ax, 10);
[Y2, T2] = step(H1bx, 10);
[Y3, T3] = step(H1cx, 10);

figure(2); clf; subplot(321);
  plot(T1, Y1, 'b-',  'Linewidth', AM_data_linewidth); hold on;
  plot(T2, Y2, 'b-', 'Linewidth', AM_data_linewidth); hold on;
  plot(T3, Y3, 'b-', 'Linewidth', AM_data_linewidth); hold on;
  plot([0 10], [1 1], 'k-', 'Linewidth', AM_ref_linewidth); hold on;

  amaxis([0, 10, -0.1, 1.4]); 
  xlabel('time');   ylabel('position');

  arcarrow([1.3 0.8], [5 0.45], -6);
  text(5.3, 0.4, 'rho');

  amprint('pvtol-lqrstep2.eps');

% Output weighting - change Qx to use outputs
Qx2 = [Cx; Cy]' * [Cx; Cy];
Qu2 = 0.1 * diag([1, 1]);
K2 = lqr(A, B, Qx2, Qu2);

H2x = ss(A-B*K2,B(:,1)*K2(1,:)*xd,Cx,0);
H2y = ss(A-B*K2,B(:,2)*K2(2,:)*yd,Cy,0);

figure(3); step(H2x, H2y, 10);
legend('x', 'y');

%%
%% Physically motivated weighting
%%
%% Shoot for 1 cm error in x, 10 cm error in y.  Try to keep the angle
%% less than 5 degrees in making the adjustments.  Penalize side forces
%% due to loss in efficiency.
%%

Qx3 = diag([100, 10, 2*pi/5, 0, 0, 0]);
Qu3 = 0.1 * diag([1, 10]);
K3 = lqr(A, B, Qx3, Qu3);

H3x = ss(A-B*K3,B(:,1)*K3(1,:)*xd,Cx,0);
H3y = ss(A-B*K3,B(:,2)*K3(2,:)*yd,Cy,0);
figure(4);  clf; subplot(221);
step(H3x, H3y, 10);
legend('x', 'y');

%%
%% Velocity control
%%
%% In this example, we modify the system so that we control the
%% velocity of the system in the x direction.  We ignore the 
%% dynamics in the vertical (y) direction.  These dynamics demonstrate
%% the role of the feedforward system since the equilibrium point 
%% corresponding to vd neq 0 requires a nonzero input.
%%
%% For this example, we use a control law u = -K(x-xd) + ud and convert 
%% this to the form u = -K x + N r, where r is the reference input and
%% N is computed as described in class.
%%

% Extract system dynamics: theta, xdot, thdot
Av = A([3 4 6], [3 4 6]);
Bv = B([3 4 6], 1);
Cv = [0 1 0];				% choose vx as output
Dv = 0;	

% Design the feedback term using LQR
Qxv = diag([2*pi/5, 10, 0]);
Quv = 0.1;
Kv = lqr(Av, Bv, Qxv, Quv);

% Design the feedforward term by solve for eq pt in terms of reference r
T = [Av Bv; Cv Dv];			% system matrix
Nxu = T \ [0; 0; 0; 1];			% compute [Nx; Nu]
Nx = Nxu(1:3); Nu = Nxu(4);		% extract Nx and Nu
N = Nu + Kv*Nx;				% compute feedforward term

%%
%% Design #1: no feedforward input, ud
%%

Nv1 = [0; 1; 0];
Hv1 = ss(Av-Bv*Kv, Bv*Kv*Nx, Cv, 0);
step(Hv1, 10);

%%
%% Design #2: compute feedforward gain corresponding to equilibrium point
%%

Hv2 = ss(Av-Bv*Kv, Bv*N, Cv, 0);
step(Hv2, 10);

%%
%% Design #3: integral action
%%
%% Add a new state to the system that is given by xidot = v - vd.  We
%% construct the control law by computing an LQR gain for the augmented
%% system.
%%

Ai = [Av, [0; 0; 0]; [Cv, 0]];
Bi = [Bv; 0];
Ci = [Cv, 0];
Di = Dv;

% Design the feedback term, including weight on integrator error
Qxi = diag([2*pi/5, 10, 0, 10]);
Qui = 0.1;
Ki = lqr(Ai, Bi, Qxi, Qui);

% Desired state (augmented)
xid = [0; 1; 0; 0];

% Construct the closed loop system (including integrator)
Hi = ss(Ai-Bi*Ki,Bi*Ki*xid - [0; 0; 0; Ci*xid],Ci,0);
step(Hi, 10);

