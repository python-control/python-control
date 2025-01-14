# repr-galler.py - different system representations for comparing versions
# RMM, 30 Dec 2024
#
# This file creates different types of systems and generates a variety
# of representations (__repr__, __str__) for those systems that can be
# used to compare different versions of python-control.  It is mainly
# intended for uses by developers to make sure there are no unexpected
# changes in representation formats, but also has some interesting
# examples of different choices in system representation.

import numpy as np

import control as ct
import control.flatsys as fs

#
# Create systems of different types
#
syslist = []

# State space (continuous and discrete time)
sys_ss = ct.ss([[0, 1], [-4, -5]], [0, 1], [-1, 1], 0, name='sys_ss')
sys_dss = sys_ss.sample(0.1, name='sys_dss')
sys_ss0 = ct.ss([], [], [], np.eye(2), name='stateless', inputs=['u0', 'u1'])
syslist += [sys_ss, sys_dss, sys_ss0]

# Transfer function (continuous and discrete time)
sys_tf = ct.tf(sys_ss)
sys_dtf = ct.tf(sys_dss, name='sys_dss_poly', display_format='poly')
sys_gtf = ct.tf([1], [1, 0])
syslist += [sys_tf, sys_dtf, sys_gtf]

# MIMO transfer function (continuous time only)
sys_mtf = ct.tf(
    [[sys_tf.num[0][0].tolist(), [0]], [[1, 0], [1, 0]  ]],
    [[sys_tf.den[0][0].tolist(), [1]], [[1],    [1, 2, 1]]],
    name='sys_mtf_zpk', display_format='zpk')
syslist += [sys_mtf]

# Frequency response data (FRD) system (continuous and discrete time)
sys_frd = ct.frd(sys_tf, np.logspace(-1, 1, 5))
sys_dfrd = ct.frd(sys_dtf, np.logspace(-1, 1, 5))
sys_mfrd = ct.frd(sys_mtf, np.logspace(-1, 1, 5))
syslist += [sys_frd, sys_dfrd, sys_mfrd]

# Nonlinear system (with linear dynamics), continuous time
def nl_update(t, x, u, params):
    return sys_ss.A @ x + sys_ss.B @ u

def nl_output(t, x, u, params):
    return sys_ss.C @ x + sys_ss.D @ u

nl_params = {'a': 0, 'b': 1}

sys_nl = ct.nlsys(
    nl_update, nl_output, name='sys_nl', params=nl_params,
    states=sys_ss.nstates, inputs=sys_ss.ninputs, outputs=sys_ss.noutputs)

# Nonlinear system (with linear dynamics), discrete time
def dnl_update(t, x, u, params):
    return sys_ss.A @ x + sys_ss.B @ u

def dnl_output(t, x, u, params):
    return sys_ss.C @ x + sys_ss.D @ u

sys_dnl = ct.nlsys(
    dnl_update, dnl_output, dt=0.1, name='sys_dnl',
    states=sys_ss.nstates, inputs=sys_ss.ninputs, outputs=sys_ss.noutputs)

syslist += [sys_nl, sys_dnl]

# Interconnected system
proc = ct.ss([[0, 1], [-4, -5]], np.eye(2), [[-1, 1], [1, 0]], 0, name='proc')
ctrl = ct.ss([], [], [], [[-2, 0], [0, -3]], name='ctrl')

proc_nl = ct.nlsys(proc, name='proc_nl')
ctrl_nl = ct.nlsys(ctrl, name='ctrl_nl')
sys_ic = ct.interconnect(
    [proc_nl, ctrl_nl], name='sys_ic',
    connections=[['proc_nl.u', 'ctrl_nl.y'], ['ctrl_nl.u', '-proc_nl.y']],
    inplist=['ctrl_nl.u'], inputs=['r[0]', 'r[1]'],
    outlist=['proc_nl.y'], outputs=proc_nl.output_labels)
syslist += [sys_ic]

# Linear interconnected system
sys_lic = ct.interconnect(
    [proc, ctrl], name='sys_ic',
    connections=[['proc.u', 'ctrl.y'], ['ctrl.u', '-proc.y']],
    inplist=['ctrl.u'], inputs=['r[0]', 'r[1]'],
    outlist=['proc.y'], outputs=proc.output_labels)
syslist += [sys_lic]

# Differentially flat system (with implicit dynamics), continuous time (only)
def fs_forward(x, u):
    return np.array([x[0], x[1], -4 * x[0] - 5 * x[1] + u[0]])

def fs_reverse(zflag):
    return (
        np.array([zflag[0][0], zflag[0][1]]),
        np.array([4 * zflag[0][0] + 5 * zflag[0][1] + zflag[0][2]]))

sys_fs = fs.flatsys(
    fs_forward, fs_reverse, name='sys_fs',
    states=sys_nl.nstates, inputs=sys_nl.ninputs, outputs=sys_nl.noutputs)

# Differentially flat system (with nonlinear dynamics), continuous time (only)
sys_fsnl = fs.flatsys(
    fs_forward, fs_reverse, nl_update, nl_output, name='sys_fsnl',
    states=sys_nl.nstates, inputs=sys_nl.ninputs, outputs=sys_nl.noutputs)

syslist += [sys_fs, sys_fsnl]

# Utility function to display outputs
def display_representations(
        description, fcn, class_list=(ct.InputOutputSystem, )):
    print("=" * 76)
    print(" " * round((76 - len(description)) / 2) + f"{description}")
    print("=" * 76 + "\n")
    for sys in syslist:
        if isinstance(sys, tuple(class_list)):
            print(str := f"{type(sys).__name__}: {sys.name}, dt={sys.dt}:")
            print("-" * len(str))
            print(fcn(sys))
            print("----\n")

# Default formats
display_representations("Default repr", repr)
display_representations("Default str (print)", str)

# 'info' format (if it exists and hasn't already been displayed)
if getattr(ct.InputOutputSystem, '_repr_info_', None) and \
   ct.config.defaults.get('iosys.repr_format', None) and \
   ct.config.defaults['iosys.repr_format'] != 'info':
    with ct.config.defaults({'iosys.repr_format': 'info'}):
        display_representations("repr_format='info'", repr)

# 'eval' format (if it exists and hasn't already been displayed)
if getattr(ct.InputOutputSystem, '_repr_eval_', None) and \
   ct.config.defaults.get('iosys.repr_format', None) and \
   ct.config.defaults['iosys.repr_format'] != 'eval':
    with ct.config.defaults({'iosys.repr_format': 'eval'}):
        display_representations("repr_format='eval'", repr)

# Change the way counts are displayed
with ct.config.defaults(
        {'iosys.repr_show_count':
         not ct.config.defaults['iosys.repr_show_count']}):
    display_representations(
        f"iosys.repr_show_count={ct.config.defaults['iosys.repr_show_count']}",
        repr, class_list=[ct.StateSpace])

# ZPK format for transfer functions
with ct.config.defaults({'xferfcn.display_format': 'zpk'}):
    display_representations(
        "xferfcn.display_format=zpk, str (print)", str,
        class_list=[ct.TransferFunction])
