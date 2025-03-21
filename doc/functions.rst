.. _function-ref:

******************
Function Reference
******************

.. Include header information from the main control module
.. automodule:: control
   :no-members:
   :no-inherited-members:
   :no-special-members:


System Creation
===============

Functions that create input/output systems from a description of the
system properties:

.. autosummary::
   :toctree: generated/

    ss
    tf
    frd
    nlsys
    zpk
    pade
    rss
    drss

Functions that transform systems from one form to another:

.. autosummary::
   :toctree: generated/

    canonical_form
    modal_form
    observable_form
    reachable_form
    sample_system
    similarity_transform
    ss2tf
    tf2ss
    tfdata

.. _interconnections-ref:

System Interconnections
=======================

.. autosummary::
   :toctree: generated/

    series
    parallel
    negate
    feedback
    interconnect
    append
    combine_tf
    split_tf
    summing_junction
    connection_table
    combine_tf
    split_tf


Time Response
=============

.. autosummary::
   :toctree: generated/

    forced_response
    impulse_response
    initial_response
    input_output_response
    step_response
    time_response_plot
    combine_time_responses


Phase plane plots
-----------------

.. automodule:: control.phaseplot
   :no-members:
   :no-inherited-members:
   :no-special-members:

.. Reset current module to main package to force reference to use prefix
.. currentmodule:: control

.. autosummary::
   :toctree: generated/

    phase_plane_plot
    phaseplot.boxgrid
    phaseplot.circlegrid
    phaseplot.equilpoints
    phaseplot.meshgrid
    phaseplot.separatrices
    phaseplot.streamlines
    phaseplot.vectorfield
    phaseplot.streamplot


Frequency Response
==================

.. autosummary::
   :toctree: generated/

    bode_plot
    describing_function_plot
    describing_function_response
    frequency_response
    nyquist_response
    nyquist_plot
    gangof4_response
    gangof4_plot
    nichols_plot
    nichols_grid


Control System Analysis
=======================

Time domain analysis:

.. autosummary::
   :toctree: generated/

    damp
    step_info

Frequency domain analysis:

.. autosummary::
   :toctree: generated/

    bandwidth
    dcgain
    linfnorm
    margin
    stability_margins
    system_norm
    phase_crossover_frequencies
    singular_values_plot
    singular_values_response
    sisotool

Pole/zero-based analysis:

.. autosummary::
   :toctree: generated/

    poles
    zeros
    pole_zero_map
    pole_zero_plot
    pole_zero_subplots
    root_locus_map
    root_locus_plot

Passive systems analysis:

.. autosummary::
   :toctree: generated/

    get_input_ff_index
    get_output_fb_index
    ispassive
    solve_passivity_LMI


Control System Synthesis
========================

State space synthesis:

.. autosummary::
   :toctree: generated/

    create_statefbk_iosystem
    dlqr
    lqr
    place
    place_acker
    place_varga

Frequency domain synthesis:

.. autosummary::
   :toctree: generated/

    h2syn
    hinfsyn
    mixsyn
    rootlocus_pid_designer


System ID and Model Reduction
=============================
.. autosummary::
   :toctree: generated/

    minimal_realization
    balanced_reduction
    hankel_singular_values
    model_reduction
    eigensys_realization
    markov
    observer_kalman_identification

Nonlinear System Support
========================
.. autosummary::
   :toctree: generated/

    find_operating_point
    linearize


Describing functions
--------------------
.. autosummary::
   :toctree: generated/

   describing_function
   friction_backlash_nonlinearity
   relay_hysteresis_nonlinearity
   saturation_nonlinearity


Differentially flat systems
---------------------------
.. automodule:: control.flatsys
   :no-members:
   :no-inherited-members:
   :no-special-members:

.. Reset current module to main package to force reference to use prefix
.. currentmodule:: control

.. autosummary::
   :toctree: generated/

   flatsys.flatsys
   flatsys.point_to_point
   flatsys.solve_flat_optimal


Optimal control
---------------
.. automodule:: control.optimal
   :no-members:
   :no-inherited-members:
   :no-special-members:

.. Reset current module to main package to force reference to use prefix
.. currentmodule:: control

.. autosummary::
   :toctree: generated/

   optimal.create_mpc_iosystem
   optimal.disturbance_range_constraint
   optimal.gaussian_likelihood_cost
   optimal.input_poly_constraint
   optimal.input_range_constraint
   optimal.output_poly_constraint
   optimal.output_range_constraint
   optimal.quadratic_cost
   optimal.solve_optimal_trajectory
   optimal.solve_optimal_estimate
   optimal.state_poly_constraint
   optimal.state_range_constraint


Stochastic System Support
=========================
.. autosummary::
   :toctree: generated/

    correlation
    create_estimator_iosystem
    dlqe
    lqe
    white_noise


Matrix Computations
===================
.. autosummary::
   :toctree: generated/

    care
    ctrb
    dare
    dlyap
    lyap
    obsv
    gram

.. _utility-and-conversions:

Utility Functions
=================
.. autosummary::
   :toctree: generated/

    augw
    bdschur
    db2mag
    isctime
    isdtime
    iosys_repr
    issiso
    mag2db
    reset_defaults
    reset_rcParams
    set_defaults
    ssdata
    timebase
    unwrap
    use_fbs_defaults
    use_legacy_defaults
    use_matlab_defaults
