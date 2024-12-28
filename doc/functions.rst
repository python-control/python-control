.. _function-ref:

******************
Function reference
******************

.. Include header information from the main control module
.. automodule:: control
   :no-members:
   :no-inherited-members:
   :no-special-members:

System creation
===============

The control toolbox makes use of "factory functions" to create input/output
systems of different types (classes):

.. autosummary::
   :toctree: generated/

    ss
    tf
    frd
    nlsys
    zpk
    rss
    drss

Systems can also be created by transforming existing systems:

.. autosummary::
   :toctree: generated/

    canonical_form
    modal_form
    observable_form
    reachable_form
    similarity_transform
    pade
    ss2tf
    tf2ss
    tfdata

.. _interconnections-ref:

System interconnections
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
    connection_table
    combine_tf
    split_tf

Time domain simulation
======================

.. autosummary::
   :toctree: generated/

    forced_response
    impulse_response
    initial_response
    input_output_response
    phase_plane_plot
    step_response
    time_response_plot
    combine_time_responses
    phaseplot.boxgrid
    phaseplot.circlegrid
    phaseplot.equilpoints
    phaseplot.meshgrid
    phaseplot.separatrices
    phaseplot.streamlines
    phaseplot.vectorfield


Frequency response
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

Control system analysis
=======================
.. autosummary::
   :toctree: generated/

    bandwidth
    damp
    dcgain
    describing_function
    get_input_ff_index
    get_output_fb_index
    ispassive
    margin
    norm
    solve_passivity_LMI
    stability_margins
    step_info
    phase_crossover_frequencies
    poles
    zeros
    pole_zero_map
    pole_zero_plot
    pole_zero_subplots
    root_locus_map
    root_locus_plot
    singular_values_plot
    singular_values_response
    sisotool
    StateSpace.__call__
    TransferFunction.__call__

Control system synthesis
========================
.. autosummary::
   :toctree: generated/

    acker
    create_statefbk_iosystem
    create_estimator_iosystem
    dlqr
    h2syn
    hinfsyn
    lqr
    mixsyn
    place
    place_varga
    rootlocus_pid_designer

System ID and model reduction
=============================
.. autosummary::
   :toctree: generated/

    minimal_realization
    balanced_reduction
    hankel_singular_values
    model_reduction
    eigensys_realization
    markov

Nonlinear system support
========================
.. autosummary::
   :toctree: generated/

    describing_function
    find_operating_point
    linearize
    input_output_response
    summing_junction
    flatsys.point_to_point

Stochastic system support
=========================
.. autosummary::
   :toctree: generated/

    correlation
    create_estimator_iosystem
    dlqe
    lqe
    white_noise

Optimal control
===============
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
   optimal.solve_ocp
   optimal.solve_oep
   optimal.state_poly_constraint
   optimal.state_range_constraint


Describing functions
====================
.. autosummary::
   :toctree: generated/

   friction_backlash_nonlinearity
   relay_hysteresis_nonlinearity
   saturation_nonlinearity

Differentially flat systems
===========================
.. autosummary::
   :toctree: generated/

   flatsys.flatsys
   flatsys.point_to_point
   flatsys.solve_flat_ocp

Matrix computations
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

Utility functions
=================
.. autosummary::
   :toctree: generated/

    augw
    bdschur
    db2mag
    isctime
    isdtime
    issiso
    mag2db
    reset_defaults
    reset_rcParams
    sample_system
    set_defaults
    ssdata
    timebase
    unwrap
    use_fbs_defaults
    use_legacy_defaults
    use_matlab_defaults
