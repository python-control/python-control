from control import tf
from control.matlab import rss
from numpy import logspace
from timeit import timeit


if __name__ == '__main__':
    nstates = 10
    sys = rss(nstates)
    sys_tf = tf(sys)
    w = logspace(-1,1,50)
    ntimes = 1000
    time_ss = timeit("sys.freqquency_response(w)", setup="from __main__ import sys, w", number=ntimes)
    time_tf = timeit("sys_tf.frequency_response(w)", setup="from __main__ import sys_tf, w", number=ntimes)
    print("State-space model on %d states: %f" % (nstates, time_ss))
    print("Transfer-function model on %d states: %f" % (nstates, time_tf))
