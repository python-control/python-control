.. _matlab-module:

****************************
 MATLAB Compatibility Module
****************************

.. automodule:: control.matlab
   :no-members:
   :no-inherited-members:
   :no-special-members:

.. warning:: This module is not closely maintained and some functionality
             in main control package may not be be available via the MATLAB
             compatibility module.

Creating Linear Models
======================
.. autosummary::
   :toctree: generated/

   tf
   ss
   frd
   zpk

Utility Functions and Conversions
=================================
.. autosummary::
   :toctree: generated/

   mag2db
   db2mag
   c2d
   ss2tf
   tf2ss
   tfdata

System Interconnections
=======================
.. autosummary::
   :toctree: generated/

   series
   parallel
   feedback
   negate
   connect
   append

System Gain and Dynamics
========================
.. autosummary::
   :toctree: generated/

   dcgain
   pole
   zero
   damp
   pzmap

Time-Domain Analysis
====================
.. autosummary::
   :toctree: generated/

   step
   impulse
   initial
   lsim

Frequency-Domain Analysis
=========================
.. autosummary::
   :toctree: generated/

   bode
   nyquist
   nichols
   margin
   freqresp
   evalfr

Compensator Design
==================
.. autosummary::
   :toctree: generated/

   rlocus
   sisotool
   place
   lqr
   dlqr
   lqe
   dlqe

State-space (SS) Models
=======================
.. autosummary::
   :toctree: generated/

   rss
   drss
   ctrb
   obsv
   gram

Model Simplification
====================
.. autosummary::
   :toctree: generated/

   minreal
   hsvd
   balred
   modred
   era
   markov

Time Delays
===========
.. autosummary::
   :toctree: generated/

   pade

Matrix Equation Solvers and Linear Algebra
==========================================
.. autosummary::
   :toctree: generated/

   lyap
   dlyap
   care
   dare

Additional Functions
====================
.. autosummary::
   :toctree: generated/

   gangof4
   unwrap

Functions Imported from Other Packages
======================================
.. autosummary::

   ~numpy.linspace
   ~numpy.logspace
   ~scipy.signal.ss2zpk
   ~scipy.signal.tf2zpk
   ~scipy.signal.zpk2ss
   ~scipy.signal.zpk2tf
