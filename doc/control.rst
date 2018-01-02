.. _function-ref:

==================
Function reference
==================

.. automodule:: control
    :no-members:
    :no-inherited-members:

.. currentmodule:: control

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

Frequency domain plotting
=========================

.. autosummary::
    :toctree: generated/

    bode_plot
    nyquist_plot
    gangof4_plot
    nichols_plot

Time domain simulation
======================

.. autosummary::
    :toctree: generated/

    forced_response
    impulse_response
    initial_response
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

.. _utility-and-conversions:

Utility functions and conversions
=================================
.. autosummary::
    :toctree: generated/

    unwrap
    db2mag
    mag2db
    damp
    isctime
    isdtime
    issiso
    issys
    pade
    sample_system
    canonical_form
    observable_form
    reachable_form
    ss2tf
    ssdata
    tf2ss
    tfdata
    timebase
    timebaseEqual
