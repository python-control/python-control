% -*-octave-*-
% octave test file
sys = ss([-2 0; 0 -1], eye(2), eye(2), zeros(2,2))

% convert to frd, 1 frequency
fr = frd(sys, [1])

% the matching matrix
frm = [ 0.4-0.2i 0;
        0     0.5-0.5i]

% feedback of the system itself
sys2 = feedback(sys, [0 1; 3 0])

% and the matching fr
fr2 = frd(sys2, [1])

fr2b = feedback(fr, [0 1; 3 0])

% frequency response from the matrix should be
frm*inv(eye(2)+[0 1; 3 0]*frm)

% one with 3 out and 2 inputs, 3 states
bsys = ss([-2 0 0; 0 -1 1; 0 0 -3], [1 0; 0 0; 0 1], eye(3), zeros(3,2))

% convert to frd, 1 frequency
bfr = frd(bsys, [1])

% the matching matrix
bfrm = [ 0.4-0.2i 0;
	 0     0.1-0.2i;
        0  0.3-0.1i]

K = [1 0.3 0; 0.1 0 0]
bsys2 = feedback(bsys, K)

% and the matching fr
bfr2 = frd(bsys2, [1])

bfr2b = feedback(bfr, K)

% frequency response from the matrix should be
bfrm*inv(eye(2)+K*bfrm)
