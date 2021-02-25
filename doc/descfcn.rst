.. _descfcn-module:

********************
Describing functions
********************

For nonlinear systems consisting of a feedback connection between a
linear system and a static nonlinearity, it is possible to obtain a
generalization of Nyquist's stability criterion based on the idea of
describing functions.  The basic concept involves approximating the
response of a static nonlinearity to an input :math:`u = A e^{j \omega
t}` as an output :math:`y = N(A) (A e^{j \omega t})`, where :math:`N(A)
\in \mathbb{C}` represents the (amplitude-dependent) gain and phase
associated with the nonlinearity.

Stability analysis of a linear system :math:`H(s)` with a feedback
nonlinearity :math:`F(x)` is done by looking for amplitudes :math:`A`
and frequencies :math:`\omega` such that

.. math::

   H(j\omega) N(A) = -1

If such an intersection exists, it indicates that there may be a limit
cycle of amplitude :math:`A` with frequency :math:`\omega`.

Describing function analysis is a simple method, but it is approximate
because it assumes that higher harmonics can be neglected. 

Module usage
============

The function :func:`~control.describing_function` can be used to
compute the describing function of a nonlinear function::

  N = ct.describing_function(F, A)

Stability analysis using describing functions is done by looking for
amplitudes :math:`a` and frequencies :math`\omega` such that

.. math::

   H(j\omega) = \frac{-1}{N(A)}

These points can be determined by generating a Nyquist plot in which the
transfer function :math:`H(j\omega)` intersections the negative
reciprocal of the describing function :math:`N(A)`.  The
:func:`~control.describing_function_plot` function generates this plot
and returns the amplitude and frequency of any points of intersection::

    ct.describing_function_plot(H, F, amp_range[, omega_range])


Pre-defined nonlinearities
==========================

To facilitate the use of common describing functions, the following
nonlinearity constructors are predefined:

.. code:: python

  friction_backlash_nonlinearity(b)	# backlash nonlinearity with width b
  relay_hysteresis_nonlinearity(b, c)   # relay output of amplitude b with
					# hysteresis of half-width c
  saturation_nonlinearity(ub[, lb])	# saturation nonlinearity with upper
					# bound and (optional) lower bound

Calling these functions will create an object `F` that can be used for
describing function analysis.  For example, to create a saturation
nonlinearity::

  F = ct.saturation_nonlinearity(1)

These functions use the
:class:`~control.DescribingFunctionNonlinearity`, which allows an
analytical description of the describing function.

Module classes and functions
============================
.. autosummary::
   :toctree: generated/

   ~control.DescribingFunctionNonlinearity
   ~control.friction_backlash_nonlinearity
   ~control.relay_hysteresis_nonlinearity
   ~control.saturation_nonlinearity
