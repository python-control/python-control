.. currentmodule:: control

.. _stochastic-systems:

******************
Stochastic Systems
******************

The Python Control Systems Library has support for basic operations
involving linear and nonlinear I/O systems with Gaussian white noise
as an input.


Stochastic signals
==================

A stochastic signal is a representation of the output of a random
process.  NumPy and SciPy have a functions to calculate the covariance
and correlation of random signals:

  * :func:`numpy.cov` - with a single argument, returns the sample
    variance of a vector random variable :math:`X \in \mathbb{R}^n`
    where the input argument represents samples of :math:`X`.  With
    two arguments, returns the (cross-)covariance of random variables
    :math:`X` and :math:`Y` where the input arguments represent
    samples of the given random variables.

  * :func:`scipy.signal.correlate` - the "cross-correlation" between two
    random (1D) sequences.  If these sequences came from a random
    process, this is a single sample approximation of the (discrete
    time) correlation function.  Use the function
    :func:`scipy.signal.correlation_lags` to compute the lag
    :math:`\tau` and :func:`scipy.signal.correlate` to get the (auto)
    correlation function :math:`r_X(\tau)`.

The python-control package has variants of these functions that do
appropriate processing for continuous time models.

The :func:`white_noise` function generates a (multi-variable) white
noise signal of specified intensity as either a sampled continuous time
signal or a discrete time signal.  A white noise signal along a 1D
array of linearly spaced set of times `timepts` can be computing using

.. code::

  V = ct.white_noise(timepts, Q[, dt])

where `Q` is a positive definite matrix providing the noise
intensity and `dt` is the sampling time (or 0 for continuous time).

In continuous time, the white noise signal is scaled such that the
integral of the covariance over a sample period is `Q`, thus
approximating a white noise signal.  In discrete time, the white noise
signal has covariance `Q` at each point in time (without any
scaling based on the sample time).

The python-control :func:`correlation` function computes the
correlation matrix :math:`{\mathbb E}\{X^\mathsf{T}(t+\tau) X(t)\}` or the
cross-correlation matrix :math:`{\mathbb E}\{X^\mathsf{T}(t+\tau) Y(t)\}`:

.. code::

  tau, Rtau = ct.correlation(timepts, X[, Y])

where :math:`\mathbb{E}` represents expectation.  The signal `X` (and
`Y`, if present) represents a continuous time signal sampled at
regularly spaced times `timepts`.  The return value provides the
correlation :math:`R_\tau` between :math:`X(t+\tau)` and :math:`X(t)`
at a set of time offsets :math:`\tau` (determined based on the spacing
of entries in the `timepts` vector.

Note that the computation of the correlation function is based on a
single time signal (or pair of time signals) and is thus a very crude
approximation to the true correlation function between two random
processes.

To compute the response of a linear (or nonlinear) system to a white
noise input, use the :func:`forced_response` (or
:func:`input_output_response`) function:

.. code::

  a, c = 1, 1
  sys = ct.ss([[-a]], [[1]], [[c]], 0)
  timepts = np.linspace(0, 5, 1000)
  Q = np.array([[0.1]])
  V = ct.white_noise(timepts, Q)
  resp = ct.forced_response(sys, timepts, V)

The correlation function for the output can be computed using the
:func:`correlation` function and compared to the analytical expression:

.. code::

  tau, r_Y = ct.correlation(timepts, resp.outputs)
  plt.plot(tau, r_Y)
  plt.plot(tau, c**2 * Q.item() / (2 * a) * np.exp(-a * np.abs(tau)))

.. _kalman-filter:

Linear quadratic estimation (Kalman filter)
===========================================

A standard application of stochastic linear systems is the computation
of the linear estimator under the assumption of white Gaussian
measurement and process noise.  This estimator is called the linear
quadratic estimator (LQE) and its gains can be computed using the
:func:`lqe` function.

We consider a continuous time, state space system

.. math::

     \frac{dx}{dt} &= Ax + Bu + Gw \\
     y &= Cx + Du + v

with unbiased process noise :math:`w` and measurement noise :math:`v`
with covariances satisfying

.. math::

   {\mathbb E}\{w w^T\} = QN,\qquad
   {\mathbb E}\{v v^T\} = RN,\qquad
   {\mathbb E}\{w v^T\} = NN

where :math:`{\mathbb E}\{\cdot\}` represents expectation.

The :func:`lqe` function computes the observer gain matrix :math:`L`
such that the stationary (non-time-varying) Kalman filter

.. math::

     \frac{d\hat x}{dt} = A \hat x + B u + L(y - C\hat x - D u),

produces a state estimate :math:`\hat x` that minimizes the expected
squared error using the sensor measurements :math:`y`.

As with the :func:`lqr` function, the :func:`lqe` function can be
called in several forms:

  * `L, P, E = lqe(sys, QN, RN)`
  * `L, P, E = lqe(sys, QN, RN, NN)`
  * `L, P, E = lqe(A, G, C, QN, RN)`
  * `L, P, E = lqe(A, G, C, QN, RN, NN)`

where `sys` is an :class:`LTI` object, and `A`, `G`, `C`, `QN`, `RN`,
and `NN` are 2D arrays of appropriate dimension.  If `sys` is a
discrete time system, the first two forms will compute the discrete
time optimal controller.  For the second two forms, the :func:`dlqr`
function can be used.  Additional arguments and details are given on
the :func:`lqr` and :func:`dlqr` documentation pages.

The :func:`create_estimator_iosystem` function can be used to create
an I/O system implementing a Kalman filter, including integration of
the Riccati ODE.  The command has the form

.. code::

  estim = ct.create_estimator_iosystem(sys, Qv, Qw)

The input to the estimator is the measured outputs `Y` and the system
input `U`.  To run the estimator on a noisy signal, use the command

.. code::

  resp = ct.input_output_response(est, timepts, [Y, U], [X0, P0])

If desired, the :func:`correct` parameter can be set to :func:`False`
to allow prediction with no additional sensor information::

  resp = ct.input_output_response(
      estim, timepts, 0, [X0, P0], param={'correct': False})

The :func:`create_statefbk_iosystem` function can be used to combine
an estimator with a state feedback controller::

  K, _, _ = ct.lqr(sys, Qx, Qu)
  estim = ct.create_estimator_iosystem(sys, Qv, Qw, P0)
  ctrl, clsys = ct.create_statefbk_iosystem(sys, K, estimator=estim)

The controller will have the same form as a full state feedback
controller, but with the system state :math:`x` input replaced by the
estimated state :math:`\hat x` (output of `estim`):

.. math::

  u = u_\text{d} - K (\hat x - x_\text{d}).

The closed loop controller `clsys` includes both the state
feedback and the estimator dynamics and takes as its input the desired
state :math:`x_\text{d}` and input :math:`u_\text{d}`::

  resp = ct.input_output_response(
      clsys, timepts, [Xd, Ud], [X0, np.zeros_like(X0), P0])


Maximum likelihood estimation
=============================

Consider a *nonlinear* system with discrete time dynamics of the form

.. math::
  :label: eq_fusion_nlsys-oep

  X[k+1] = f(X[k], u[k], V[k]), \qquad Y[k] = h(X[k]) + W[k],

where :math:`X[k] \in \mathbb{R}^n`, :math:`u[k] \in \mathbb{R}^m`, and
:math:`Y[k] \in \mathbb{R}^p`, and :math:`V[k] \in \mathbb{R}^q` and
:math:`W[k] \in \mathbb{R}^p` represent random processes that are not
necessarily Gaussian white noise processes.  The estimation problem that we
wish to solve is to find the estimate :math:`\hat x[\cdot]` that matches
the measured outputs :math:`y[\cdot]` with "likely" disturbances and
noise.

For a fixed horizon of length :math:`N`, this problem can be formulated as
an optimization problem where we define the likelihood of a given estimate
(and the resulting noise and disturbances predicted by the model) as a cost
function. Suppose we model the likelihood using a conditional probability
density function :math:`p(x[0], \dots, x[N] \mid y[0], \dots, y[N-1])`.
Then we can pose the state estimation problem as

.. math::
  :label: eq_fusion_oep

  \hat x[0], \dots, \hat x[N] =
  \arg \max_{\hat x[0], \dots, \hat x[N]}
  p(\hat x[0], \dots, \hat x[N] \mid y[0], \dots, y[N-1])

subject to the constraints given by equation :eq:`eq_fusion_nlsys-oep`.
The result of this optimization gives us the estimated state for the
previous :math:`N` steps in time, including the "current" time
:math:`x[N]`.  The basic idea is thus to compute the state estimate that is
most consistent with our model and penalize the noise and disturbances
according to how likely they are (based on the given stochastic system
model for each).

Given a solution to this fixed-horizon optimal estimation problem, we can
create an estimator for the state over all times by repeatedly applying the
optimization problem :eq:`eq_fusion_oep` over a moving horizon.  At each
time :math:`k`, we take the measurements for the last :math:`N` time steps
along with the previously estimated state at the start of the horizon,
:math:`x[k-N]` and reapply the optimization in equation
:eq:`eq_fusion_oep`.  This approach is known as a \define{moving horizon
estimator} (MHE).

The formulation for the moving horizon estimation problem is very general
and various situations can be captured using the conditional probability
function :math:`p(x[0], \dots, x[N] \mid y[0], \dots, y[N-1]`.  We start by
noting that if the disturbances are independent of the underlying states of
the system, we can write the conditional probability as

.. math::

  p \bigl(x[0], \dots, x[N] \mid y[0], \dots, y[N-1]\bigr) =
  p_{X[0]}(x[0])\, \prod_{k=0}^{N-1} p_V\bigl(y[k] - h(x[k])\bigr)\,
    p\bigl(x[k+1] \mid x[k]\bigr).

This expression can be further simplified by taking the log of the
expression and maximizing the function

.. math::
  :label: eq_fusion_log-likelihood

  \log p_{X[0]}(x[0]) + \sum_{k=0}^{N-1} \log
  p_W \bigl(y[k] - h(x[k])\bigr) + \log p_V(v[k]).

The first term represents the likelihood of the initial state, the
second term captures the likelihood of the noise signal, and the final
term captures the likelihood of the disturbances.

If we return to the case where :math:`V` and :math:`W` are modeled as
Gaussian processes, then it can be shown that maximizing equation
:eq:`eq_fusion_log-likelihood` is equivalent to solving the optimization
problem given by

.. math::
  :label: eq_fusion_oep-gaussian

  \min_{x[0], \{v[0], \dots, v[N-1]\}}
  \|x[0] - \bar x[0]\|_{P_0^{-1}} + \sum_{k=0}^{N-1}
  \|y[k] - h(x_k)\|_{R_W^{-1}}^2 +
  \|v[k] \|_{R_V^{-1}}^2,

where :math:`P_0`, :math:`R_V`, and :math:`R_W` are the covariances of the
initial state, disturbances, and measurement noise.

Note that while the optimization is carried out only over the estimated
initial state :math:`\hat x[0]`, the entire history of estimated states can
be reconstructed using the system dynamics:

.. math::

  \hat x[k+1] = f(\hat x[k], u[k], v[k]), \quad k = 0, \dots, N-1.

In particular, we can obtain the estimated state at the end of the moving
horizon window, corresponding to the current time, and we can thus
implement an estimator by repeatedly solving the optimization of a window
of length :math:`N` backwards in time.

The :mod:`optimal` module described in the :ref:`optimal-module`
section implements functions for solving optimal estimation problems
using maximum likelihood estimation.  The
:class:`optimal.OptimalEstimationProblem` class is used to define an
optimal estimation problem over a finite horizon::

  oep = opt.OptimalEstimationProblem(sys, timepts, cost[, constraints])

Given noisy measurements :math:`y` and control inputs :math:`u`, an
estimate of the states over the time points can be computed using the
:func:`optimal.OptimalEstimationProblem.compute_estimate` method::

  estim = oep.compute_optimal(Y, U[, X0=x0, initial_guess=(xhat, v)])
  xhat, v, w = estim.states, estim.inputs, estim.outputs

For discrete time systems, the
:func:`optimal.OptimalEstimationProblem.create_mhe_iosystem` method
can be used to generate an input/output system that implements a
moving horizon estimator.

Several functions are available to help set up standard optimal estimation
problems:

.. autosummary::

   optimal.gaussian_likelihood_cost
   optimal.disturbance_range_constraint
