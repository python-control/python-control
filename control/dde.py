import numpy as np

from scipy.integrate import LSODA


def dde_response(delay_sys, T, U=0, X0=0, params=None,
                 transpose=False, return_x=False, squeeze=None,
                 method=None):
    
    from .timeresp import TimeResponseData, _check_convert_array
    from .delaylti import DelayLTI
    if not isinstance(delay_sys, DelayLTI):
         raise TypeError("Input must be a DelayLTI")

    A, B1, B2 = delay_sys.A, delay_sys.B1, delay_sys.B2
    C1, C2 = delay_sys.C1, delay_sys.C2
    D11, D12 = delay_sys.D11, delay_sys.D12
    D21, D22 = delay_sys.D21, delay_sys.D22
    tau = delay_sys.tau

    n_states = A.shape[0]
    n_inputs = B1.shape[1]  # External inputs u
    n_outputs = C1.shape[0] # External outputs y
    n_internal_outputs = C2.shape[0] # 'z' outputs feeding delays
    n_internal_inputs = B2.shape[1]  # 'w' inputs from delays

    if U is not None:
        U = np.asarray(U)
    if T is not None:
        T = np.asarray(T)

    T = _check_convert_array(T, [('any',), (1, 'any')],
                                    'Parameter `T`: ', squeeze=True,
                                    transpose=transpose)

    n_steps = T.shape[0] 
    dt = (T[-1] - T[0]) / (n_steps - 1)
    if not np.allclose(np.diff(T), dt):
        raise ValueError("Parameter `T`: time values must be equally "
                        "spaced.")

    X0 = _check_convert_array(
        X0, [(n_states,), (n_states, 1)], 'Parameter `X0`: ', squeeze=True)
    
    # Test if U has correct shape and type
    legal_shapes = [(n_steps,), (1, n_steps)] if n_inputs == 1 else \
        [(n_inputs, n_steps)]
    U = _check_convert_array(U, legal_shapes,
                            'Parameter `U`: ', squeeze=False,
                            transpose=transpose)
    
    xout = np.zeros((n_states, n_steps))
    xout[:, 0] = X0
    yout = np.zeros((n_outputs, n_steps))
    tout = T

    # Solver here depending on the method
    if method == 'LSODA':
        xout, yout = Skogestad_Python_LSODA(delay_sys, dt, T, U, X0, xout, yout)
    else:
        xout, yout = Skogestad_Python_solver(delay_sys, dt, T, U, X0, xout, yout)
    
    return TimeResponseData(
        tout, yout, xout, U, 
        params=params, 
        issiso=delay_sys.issiso(),
        output_labels=delay_sys.output_labels, 
        input_labels=delay_sys.input_labels,
        state_labels=delay_sys.state_labels, 
        sysname=delay_sys.name, 
        plot_inputs=True,
        title="Forced response for " + delay_sys.name, 
        trace_types=['forced'],
        transpose=transpose, 
        return_x=return_x, 
        squeeze=squeeze
    )


def Skogestad_Python_solver(delay_sys, dt, T, U, X0, xout, yout):
    """
    Method from Skogestad-Python: https://github.com/alchemyst/Skogestad-Python/blob/master/robustcontrol/InternalDelay.py#L446
    RK integration.
    """
    dtss = [int(np.round(delay / dt)) for delay in delay_sys.tau]
    zs = []
    
    def f(t, x):
        return delay_sys.A @ x + delay_sys.B1 @ linear_interp_u(t, T, U) + delay_sys.B2 @ wf(zs, dtss)
    
    xs = [X0]
    ys = []
    for i,t in enumerate(T):
        x = xs[-1]

        y = delay_sys.C1 @ np.array(x) + delay_sys.D11 @ linear_interp_u(t, T, U) + delay_sys.D12 @ wf(zs, dtss)
        ys.append(list(y))

        z = delay_sys.C2 @ np.array(x) + delay_sys.D21 @ linear_interp_u(t, T, U) + delay_sys.D22 @ wf(zs, dtss)
        zs.append(list(z))

        # x integration
        k1 = f(t, x) * dt
        k2 = f(t + 0.5 * dt, x + 0.5 * k1) * dt
        k3 = f(t + 0.5 * dt, x + 0.5 * k2) * dt
        k4 = f(t + dt, x + k3) * dt
        dx = (k1 + k2 + k2 + k3 + k3 + k4) / 6
        x = [xi + dxi for xi, dxi in zip(x, dx)]
        xs.append(list(x))

        xout[:, i] = x
        yout[:, i] = y
    
    return xout, yout
    

def Skogestad_Python_LSODA(delay_sys, dt, T, U, X0, xout, yout):
    dtss = [int(np.round(delay / dt)) for delay in delay_sys.tau]
    zs = []
    
    def f(t, x):
        return delay_sys.A @ x + delay_sys.B1 @ linear_interp_u(t, T, U) + delay_sys.B2 @ wf(zs, dtss)
    
    solver = LSODA(f, T[0], X0, t_bound=T[-1], max_step=dt)

    xs = [X0]
    ts = [T[0]]
    while solver.status == "running":
        t = ts[-1]
        x = xs[-1]
        y = delay_sys.C1 @ np.array(x) + delay_sys.D11 @ linear_interp_u(t, T, U) + delay_sys.D12 @ wf(zs, dtss)
        z = delay_sys.C2 @ np.array(x) + delay_sys.D21 @ linear_interp_u(t, T, U) + delay_sys.D22 @ wf(zs, dtss)
        zs.append(list(z))

        solver.step()
        t = solver.t
        ts.append(t)

        x = solver.y.copy()
        xs.append(list(x))

        for it, ti in enumerate(T):
            if ts[-2] < ti <= ts[-1]:
                xi = solver.dense_output()(ti)
                xout[:, it] = xi
                yout[:, it] = delay_sys.C1 @ np.array(xi) + delay_sys.D11 @ linear_interp_u(t, T, U) + delay_sys.D12 @ wf(zs, dtss)
    
    return xout, yout


def linear_interp_u(t, T, U):
    if np.ndim(U) == 1:
        return np.array([np.interp(t, T, U)])
    elif np.ndim(U) == 0:
        print("U is a scalar !")
        return U
    else:
        return np.array([np.interp(t, T, ui) for ui in U])
    

def wf(zs, dtss):
    ws = []
    for i, dts in enumerate(dtss):
        if len(zs) <= dts:
            ws.append(0)
        elif dts == 0:
            ws.append(zs[-1][i])
        else:
            ws.append(zs[-dts][i])
    return np.array(ws)