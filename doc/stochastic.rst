******************
Stochastic Systems
******************

Optimal estimation problem setup
--------------------------------

Consider a nonlinear system with discrete time dynamics of the form

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
