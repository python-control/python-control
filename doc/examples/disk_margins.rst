Disk margin example
------------------------------------------

This example demonstrates the use of the `disk_margins` routine
to compute robust stability margins for a feedback system, i.e.,
variation in gain and phase one or more loops.  The SISO examples
are drawn from the published paper and the MIMO example is the
"spinning satellite" example from the MathWorks documentation.

Code
....
.. literalinclude:: disk_margins.py
   :language: python
   :linenos:

Notes
.....
1. The environment variable `PYCONTROL_TEST_EXAMPLES` is used for
testing to turn off plotting of the outputs.
