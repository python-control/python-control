.. _function-ref:

******************
Function reference
******************

.. Include header information from the main control module
.. automodule:: control
    :no-members:
    :no-inherited-members:

System creation
===============
.. autosummary::
   :toctree: generated/

    ss
    tf
    frd
    rss
    drss

System interconnections
=======================
.. autosummary::
   :toctree: generated/

    append
    connect
    feedback
    negate
    parallel
    series

See also the :ref:`iosys-module` module, which can be used to create and
interconnect nonlinear input/output systems.

Frequency domain plotting
=========================

.. autosummary::
   :toctree: generated/

    bode_plot
    nyquist_plot
    gangof4_plot
    nichols_plot

Note: For plotting commands that create multiple axes on the same plot, the
individual axes can be retrieved using the axes label (retrieved using the
`get_label` method for the matplotliib axes object).  The following labels
are currently defined:

* Bode plots: `control-bode-magnitude`, `control-bode-phase`
* Gang of 4 plots: `control-gangof4-s`, `control-gangof4-cs`,
  `control-gangof4-ps`, `control-gangof4-t`

Time domain simulation
======================

.. autosummary::
   :toctree: generated/

    forced_response
    impulse_response
    initial_response
    input_output_response
    step_response
    phase_plot

Block diagram algebra
=====================
.. autosummary::
   :toctree: generated/

    series
    parallel
    feedback
    negate

Control system analysis
=======================
.. autosummary::
   :toctree: generated/

    dcgain
    evalfr
    freqresp
    margin
    stability_margins
    phase_crossover_frequencies
    pole
    zero
    pzmap
    root_locus
    sisotool

Matrix computations
===================
.. autosummary::
   :toctree: generated/

    care
    dare
    lyap
    dlyap
    ctrb
    obsv
    gram

Control system synthesis
========================
.. autosummary::
   :toctree: generated/

    acker
    h2syn
    hinfsyn
    lqr
    mixsyn
    place

Model simplification tools
==========================
.. autosummary::
   :toctree: generated/

    minreal
    balred
    hsvd
    modred
    era
    markov

Nonlinear system support
========================
.. autosummary::
   :toctree: generated/

   ~iosys.find_eqpt
   ~iosys.linearize
   ~iosys.input_output_response
   ~iosys.ss2io
   ~iosys.tf2io
   flatsys.point_to_point

.. _utility-and-conversions:

Utility functions and conversions
=================================
.. autosummary::
   :toctree: generated/

    augw
    canonical_form
    damp
    db2mag
    isctime
    isdtime
    issiso
    issys
    mag2db
    observable_form
    pade
    reachable_form
    reset_defaults
    sample_system
    ss2tf
    ssdata
    tf2ss
    tfdata
    timebase
    timebaseEqual
    unwrap
    use_fbs_defaults
    use_matlab_defaults
    use_numpy_matrix
