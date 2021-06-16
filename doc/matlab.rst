.. _matlab-module:

****************************
 MATLAB compatibility module
****************************

.. automodule:: control.matlab
   :no-members:
   :no-inherited-members:
   :no-special-members:

Creating linear models
======================
.. autosummary::
   :toctree: generated/

   tf
   ss
   frd
   rss
   drss

Utility functions and conversions
=================================
.. autosummary::
   :toctree: generated/

   mag2db
   db2mag
   c2d
   ss2tf
   tf2ss
   tfdata

System interconnections
=======================
.. autosummary::
   :toctree: generated/

   series
   parallel
   feedback
   negate
   connect
   append

System gain and dynamics
========================
.. autosummary::
   :toctree: generated/

   dcgain
   pole
   zero
   damp
   pzmap

Time-domain analysis
====================
.. autosummary::
   :toctree: generated/

   step
   impulse
   initial
   lsim

Frequency-domain analysis
=========================
.. autosummary::
   :toctree: generated/

   bode
   nyquist
   nichols
   margin
   freqresp
   evalfr

Compensator design
==================
.. autosummary::
   :toctree: generated/

   rlocus
   sisotool
   place
   lqr

State-space (SS) models
=======================
.. autosummary::
   :toctree: generated/

   rss
   drss
   ctrb
   obsv
   gram

Model simplification
====================
.. autosummary::
   :toctree: generated/

   minreal
   hsvd
   balred
   modred
   era
   markov

Time delays
===========
.. autosummary::
   :toctree: generated/

   pade

Matrix equation solvers and linear algebra
==========================================
.. autosummary::
   :toctree: generated/

   lyap
   dlyap
   care
   dare

Additional functions
====================
.. autosummary::
   :toctree: generated/

   gangof4
   unwrap

Functions imported from other modules
=====================================
.. autosummary::

   ~numpy.linspace
   ~numpy.logspace
   ~scipy.signal.ss2zpk
   ~scipy.signal.tf2zpk
   ~scipy.signal.zpk2ss
   ~scipy.signal.zpk2tf
