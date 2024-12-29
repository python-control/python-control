.. currentmodule:: control

.. _package-configuration-parameters:

Package configuration parameters
================================

The python-control package can be customized to allow for different default
values for selected parameters.  This includes the ability to change the
way system names are created, set the style for various types of plots, and
determine default solvers and parameters to use in solving optimization
problems.

To set the default value of a configuration parameter, set the appropriate
element of the `config.defaults` dictionary::

  ct.config.defaults['module.parameter'] = value

The :func:`set_defaults` function can also be used to set multiple
configuration parameters at the same time::

  ct.set_defaults('module', param1=val1, param2=val2, ...]

Finally, there are also functions available to set collections of
parameters based on standard configurations:

.. autosummary::

   reset_defaults
   use_fbs_defaults
   use_matlab_defaults
   use_legacy_defaults


System creation parameters
--------------------------

.. py:data:: control.default_dt
   :type: int
   :value: 0

   Default value of `dt` when constructing new I/O systems.  If `dt`
   is not specified explicitly this value will be used.  Set to `None`
   to leave the timebase unspecified, 0 for continuous time systems,
   `True` for discrete time systems.

.. py:data:: iosys.converted_system_name_prefix
   :type: str
   :value: ''

   Prefix to add to system name when converting a system from one
   representation to another.

.. py:data:: iosys.converted_system_name_suffix
   :type: str
   :value: '$converted'

   Suffix to add to system name when converting a system from one
   representation to another.

.. py:data:: iosys.duplicate_system_name_prefix
   :type: str
   :value: ''

   Prefix to add to system name when making a copy of a system.

.. py:data:: iosys.duplicate_system_name_suffix
   :type: str
   :value: '$copy'

   Suffix to add to system name when making a copy of a system.

.. py:data:: iosys.indexed_system_name_prefix
   :type: str
   :value: ''

   Prefix to add to system name when extracting a subset of the inputs
   and outputs.

.. py:data:: iosys.indexed_system_name_suffix
   :type: str
   :value: '$indexed'

   Suffix to add to system name when extracting a subset of the inputs
   and outputs.

.. py:data:: iosys.linearized_system_name_prefix
   :type: str
   :value: ''

   Prefix to add to system name when linearizing a system using
   :func:`linearize`.

.. py:data:: iosys.linearized_system_name_suffix
   :type: str
   :value: '$linearized'

   Suffix to add to system name when linearizing a system using
   :func:`linearize`.

.. py:data:: iosys.sampled_system_name_prefix
   :type: str
   :value: ''

   Prefix to add to system name when sampling a system at a set of
   frequency points in :func:`frd`.

.. py:data:: iosys.sampled_system_name_suffix
   :type: str
   :value: '$sampled'

   Suffix to add to system name when sampling a system at a set of
   frequency points in :func:`frd`.

.. py:data:: iosys.state_name_delim
   :type: str
   :value: '_'

   Used by :func:`interconnect` to set the names of the states of the
   interconnected system. If the state names are not explicitly given,
   the states will be given names of the form
   '<subsys_name><delim><state_name>', for each subsys in syslist and
   each state_name of each subsys, where <delim> is the value of
   config.defaults['iosys.state_name_delim'].

.. py:data:: statesp.remove_useless_states
   :type: bool
   :value: False

   When creating a :class:`StateSpace` system, remove states that have no
   effect on the input-output dynamics of the system.


System display parameters
-------------------------

.. py:data:: iosys.repr_format
   :type: str
   :value: 'info'

   Set the default format used by :func:`iosys_repr` to create the
   representation of an :class:`InputOutputSystem`:

     * 'info' : <IOSystemType:sysname:[inputs]->[outputs]
     * 'eval' : system specific, loadable representation
     * 'latex' : latex representation of the object

.. py:data:: xferfcn.display_format
   :type: str
   :value: 'poly'

   Set the display format used in printing the
   :class:`TransferFunction` object:

     * 'poly': Single polynomial for numerator and denominator.
     * 'zpk': Product of factors, showing poles and zeros.

.. py:data:: xferfcn.floating_point_format
   :type: str
   :value: '.4g'

   Format to use for displaying coefficients in
   :class:`TransferFunction` objects when generating string
   representations.

.. py:data:: statesp.latex_num_format
   :type: str
   :value: '.3g'

   Format to use for displaying coefficients in :class:`StateSpace` systems
   when generating LaTeX representations.

.. py:data:: statesp.latex_repr_type
   :type: str
   :value: 'partitioned'

   Used when generating LaTeX representations of :class:`StateSpace`
   systems.  If 'partitioned', the A, B, C, D matrices are shown as
   a single, partitioned matrix; if 'separate', the matrices are
   shown separately.

.. py:data:: statesp.latex_maxsize
   :type: int
   :value: 10

   Maximum number of states plus inputs or outputs for which the LaTeX
   representation of the system dynamics will be generated.


Response parameters
-------------------

.. py:data:: control.squeeze_frequency_response
   :type: bool
   :value: None

   Set the default value of the `squeeze` parameter for
   :func:`frequency_response` and :class:`FrequencyResponseData`
   objects. If `None` then if a system is single-input, single-output
   (SISO) the outputs (and inputs) are returned as a 1D array (indexed by
   frequency), and if a system is multi-input or multi-output, then the
   outputs are returned as a 2D array (indexed by output and frequency) or
   a 3D array (indexed by output, input (or trace), and frequency).  If
   `squeeze=True`, access to the output response will remove
   single-dimensional entries from the shape of the inputs and outputs even
   if the system is not SISO. If `squeeze=False`, the output is returned as
   a 3D array (indexed by the output, input, and frequency) even if the
   system is SISO.

.. py:data:: control.squeeze_time_response
   :type: bool
   :value: None

   Set the default value of the `squeeze` parameter for
   :func:`input_output_response` and other time response objects. By
   default, if a system is single-input, single-output (SISO) then the
   outputs (and inputs) are returned as a 1D array (indexed by time) and if
   a system is multi-input or multi-output, then the outputs are returned
   as a 2D array (indexed by output and time) or a 3D array (indexed by
   output, input, and time).  If `squeeze=True`, access to the output
   response will remove single-dimensional entries from the shape of the
   inputs and outputs even if the system is not SISO. If `squeeze=False`,
   the output is returned as a 3D array (indexed by the output, input, and
   time) even if the system is SISO.

.. py:data:: forced_response.return_x
   :type: bool
   :value: False

   Determine whether :func:`forced_response` returns the values of the
   states when the :class:`TimeResponseData` object is evaluated.  The
   default value was `True` before version 0.9 and is `False` since then.


Plotting parameters
-------------------

.. py:data:: ctrlplot.rcParams
   :type: dict
   :value: {'axes.labelsize': 'small', 'axes.titlesize': 'small', 'figure.titlesize': 'medium', 'legend.fontsize': 'x-small', 'xtick.labelsize': 'small', 'ytick.labelsize': 'small'}

   Overrides the default Matplotlib parameters used for generating plots.

.. py:data:: freqplot.dB
   :type: bool
   :value: False

   If `True`, the magnitude in :func:`bode_plot` is plotted in dB
   (otherwise powers of 10).

.. py:data:: freqplot.deg
   :type: bool
   :value: True

   If `True`, the phase in :func:`bode_plot` is plotted in degrees
   (otherwise radians).

.. py:data:: freqplot.feature_periphery_decades
   :type: int
   :value: 1

   Number of decades in frequency to include on both sides of features
   (poles, zeros) when generating frequency plots.

.. py:data:: freqplot.freq_label
   :type: bool
   :value: 'Frequency [{units}]'

   Label for the frequency axis in frequency plots, with `units` set to
   either 'rad/sec' or 'Hz' when the label is created.

.. py:data:: freqplot.grid
   :type: bool
   :value: True

   Include grids for magnitude and phase in frequency plots.

.. py:data:: freqplot.Hz
   :type: bool
   :value: False

   If `True`, use Hertz for frequency response plots (otherwise rad/sec).

.. py:data:: freqplot.number_of_samples
   :type: int
   :value: 1000

   Number of frequency points to use in in frequency plots.

.. py:data:: freqplot.share_magnitude
   :type: str
   :value: 'row'

   Determine whether and how axis limits are shared between the magnitude
   variables in :func:`bode_plot`.  Can be set set to 'row' to share across
   all subplots in a row, 'col' to set across all subplots in a column, or
   `False` to allow independent limits.

.. py:data:: freqplot.share_phase
   :type: str
   :value: 'row'

   Determine whether and how axis limits are shared between the phase
   variables in :func:`bode_plot`.  Can be set set to 'row' to share across
   all subplots in a row, 'col' to set across all subplots in a column, or
   `False` to allow independent limits.

.. py:data:: freqplot.share_frequency
   :type: str
   :value: 'col'

   Determine whether and how axis limits are shared between the frequency
   axes in :func:`bode_plot`.  Can be set set to 'row' to share across all
   subplots in a row, 'col' to set across all subplots in a column, or
   `False` to allow independent limits.

.. py:data:: freqplot.title_frame
   :type: str
   :value: 'axes'

   Set the frame of reference used to center the plot title. If set to
   'axes', the horizontal position of the title will be centered relative
   to the axes.  If set to 'figure', it will be centered with respect to
   the figure (faster execution).

.. py:data:: freqplot.wrap_phase
   :type: bool
   :value: False

    If wrap_phase is `False`, then the phase will be unwrapped so that it
    is continuously increasing or decreasing.  If wrap_phase is `True` the
    phase will be restricted to the range [-180, 180) (or [:math:`-\\pi`,
    :math:`\\pi`) radians). If `wrap_phase` is specified as a float, the
    phase will be offset by 360 degrees if it falls below the specified
    value.

.. py:data:: nichols.grid
   :type: bool
   :value: True

   Set to `True` if :func:`nichols_plot` should include a Nichols-chart
   grid.

.. py:data:: nyquist.arrows
   :type: int
   :value: 2

   Specify the number of arrows for :func:`nyquist_plot`.  If an integer is
   passed. that number of equally spaced arrows will be plotted on each of
   the primary segment and the mirror image.  If a 1D array is passed, it
   should consist of a sorted list of floats between 0 and 1, indicating
   the location along the curve to plot an arrow.  If a 2D array is passed,
   the first row will be used to specify arrow locations for the primary
   curve and the second row will be used for the mirror image.

.. py:data:: nyquist.arrow_size
   :type: float
   :value: 8

   Arrowhead width and length (in display coordinates) for
   :func:`nyquist_plot`.

.. py:data:: nyquist.circle_style
   :type: dict
   :value: {'color': 'black', 'linestyle': 'dashed', 'linewidth': 1}

   Style for unit circle in :func:`nyquist_plot`.

.. py:data:: nyquist.encirclement_threshold
   :type: float
   :value: 0.05

   Define the threshold in :func:`nyquist_response` for generating a
   warning if the number of net encirclements is a non-integer value.

.. py:data:: nyquist.indent_direction
   :type: str
   :value: 'right'

   Set the direction of indentation in :func:`nyquist_response` for poles
   on the imaginary axis.  Valid values are 'right', 'left', or 'none'.

.. py:data:: nyquist.indent_points
   :type: int
   :value: 50

   Set the number of points to insert in the Nyquist contour around poles
   that are at or near the imaginary axis in :func:`nyquist_response`.

.. py:data:: nyquist.indent_radius
   :type: float
   :value: 0.0001

   Amount to indent the Nyquist contour around poles on or near the
   imaginary axis in :func:`nyquist_response`. Portions of the Nyquist plot
   corresponding to indented portions of the contour are plotted using a
   different line style.

.. py:data:: nyquist.max_curve_magnitude
   :type: float
   :value: 20

   Restrict the maximum magnitude of the Nyquist plot in
   :func:`nyquist_plot`.  Portions of the Nyquist plot whose magnitude is
   restricted are plotted using a different line style.

.. py:data:: nyquist.max_curve_offset
   :type: float
   :value: 0.02

   When plotting scaled portion of the Nyquist plot in
   :func:`nyquist_plot`, increase/decrease the magnitude by this fraction
   of the max_curve_magnitude to allow any overlaps between the primary and
   mirror curves to be avoided.

.. py:data:: nyquist.mirror_style
   :type: list of str
   :value: ['--', ':']

   Linestyles for mirror image of the Nyquist curve in
   :func:`nyquist_plot`.  The first element is used for unscaled portions
   of the Nyquist curve, the second element is used for portions that are
   scaled (using max_curve_magnitude).  If `False` then omit completely.

.. py:data:: nyquist.primary_style
   :type: list of str
   :value: ['-', '-.']

   Linestyles for primary image of the Nyquist curve in
   :func:`nyquist_plot`.  The first element is used for unscaled portions
   of the Nyquist curve, the second element is used for portions that are
   scaled (using max_curve_magnitude).

.. py:data:: nyquist.start_marker
   :type: str
   :value: 'o'

   Matplotlib marker to use to mark the starting point of the Nyquist plot
   in :func:`nyquist_plot`.

.. py:data:: nyquist.start_marker_size
   :type: float
   :value: 4

   Start marker size (in display coordinates) in :func:`nyquist_plot`.

.. py:data:: phaseplot.arrows
   :type: int
   :value: 2

   Set the default number of arrows in :func:`phase_plane_plot` and
   :func:`phaseplot.streamlines`.

.. py:data:: phaseplot.arrow_size
   :type: float
   :value: 8

   Set the default size of arrows in :func:`phase_plane_plot` and
   :func:`phaseplot.streamlines`.

.. py:data:: phaseplot.arrow_style
   :type: matplotlib patch
   :value: None

   Set the default style for arrows in :func:`phase_plane_plot` and
   :func:`phaseplot.streamlines`.  If set to `None`, defaults to

   .. code::

      mpl.patches.ArrowStyle(
          'simple', head_width=int(2 * arrow_size / 3),
          head_length=arrow_size)

.. py:data:: phaseplot.separatrices_radius
   :type: float
   :value: 0.1

   In :func:`phaseplot.separatrices`, set the offset from the equlibrium
   point to the starting point of the separatix traces, in the direction of
   the eigenvectors evaluated at that equilibrium point.

.. py:data:: pzmap.buffer_factor
   :type: float
   :value: 1.05

    The limits of the pole/zero plot generated by :func:`pole_zero_plot`
    are set based on the location features in the plot, including the
    location of poles, zeros, and local maxima of root locus curves.  The
    locations of local maxima are expanded by the buffer factor set by
    `buffer_factor`.

.. py:data:: pzmap.expansion_factor
   :type: float
   :value: 1.8

    The final axis limits of the pole/zero plot generated by
    :func:`pole_zero_plot` are set to by the largest features in the plot
    multiplied by an expansion factor set by `expansion_factor`.

.. py:data:: pzmap.grid
   :type: bool
   :value: False

   If `True` plot omega-damping grid in :func:`pole_zero_plot`. If `False`
   or None show imaginary axis for continuous time systems, unit circle for
   discrete time systems.  If `empty`, do not draw any additonal lines.

   Note: this setting only applies to pole/zero plots.  For root locus
   plots, the 'rlocus.grid' parameter value is used as teh default.

.. py:data:: pzmap.marker_size
   :type: float
   :value: 6

   Set the size of the markers used for poles and zeros in
   :func:`pole_zero_plot`.

.. py:data:: pzmap.marker_width
   :type: float
   :value: 1.5

   Set the line width of the markers used for poles and zeros in
   :func:`pole_zero_plot`.

.. py:data:: rlocus.grid
   :type: bool
   :value: True

   If `True` plot omega-damping grid in :func:`root_locus_plot`. If `False`
   or None show imaginary axis for continuous time systems, unit circle for
   discrete time systems.  If `empty`, do not draw any additonal lines.

.. py:data:: sisotool.initial_gain
   :type: float
   :value: 1

   Initial gain to use for plotting root locus in :func:`sisotool`.

.. py:data:: timeplot.input_props
   :type: list of dict
   :value: [{'color': 'tab:red'}, {'color': 'tab:purple'}, {'color': 'tab:brown'},  {'color': 'tab:olive'}, {'color': 'tab:cyan'}]

   List of line properties to use when plotting combined inputs in
   :func:`time_response_plot`.  The line properties for each input will be
   cycled through this list.

.. py:data:: timeplot.output_props
   :type: list of dict
   :value: [{'color': 'tab:blue'}, {'color': 'tab:orange'}, {'color': 'tab:green'}, {'color': 'tab:pink'}, {'color': 'tab:gray'}]

   List of line properties to use when plotting combined outputs in
   :func:`time_response_plot`.  The line properties for each input will be
   cycled through this list.

.. py:data:: timeplot.trace_props
   :type: list of dict
   :value: [{'linestyle': '-'}, {'linestyle': '--'}, {'linestyle': ':'}, {'linestyle': '-.'}]

   List of line properties to use when plotting multiple traces in
   :func:`time_response_plot`.  The line properties for each input will be
   cycled through this list.

.. py:data:: timeplot.sharex
   :type: str
   :value: 'col'

   Determine whether and how x-axis limits are shared between subplots in
   :func:`time_response_plot`.  Can be set set to 'row' to share across all
   subplots in a row, 'col' to set across all subplots in a column, 'all'
   to share across all subplots, or `False` to allow independent limits.

.. py:data:: timeplot.sharey
   :type: bool
   :value: False

   Determine whether and how y-axis limits are shared between subplots in
   :func:`time_response_plot`.  Can be set set to 'row' to share across all
   subplots in a row, 'col' to set across all subplots in a column, 'all'
   to share across all subplots, or `False` to allow independent limits.

.. py:data:: timeplot.time_label
   :type: str
   :value: 'Time [s]'

   Label to use for the time axis in :func:`time_response_plot`.


Optimization parameters
-----------------------

.. py:data:: optimal.minimize_method
   :type: str
   :value: None

   Set the method used by :func:`scipy.optimize.minimize` when called in
   :func:`solve_ocp` and :func:`solve_oep`.

.. py:data:: optimal.minimize_options
   :type: dict
   :value: {}

   Set the value of the options keyword used by
   :func:`scipy.optimize.minimize` when called in :func:`solve_ocp` and
   :func:`solve_oep`.

.. py:data:: optimal.minimize_kwargs
   :type: dict
   :value: {}

   Set the keyword arguments passed to :func:`scipy.optimize.minimize` when
   called in :func:`solve_ocp` and :func:`solve_oep`.

.. py:data:: optimal.solve_ivp_method
   :type: str
   :value: None

   Set the method used by :func:`scipy.integrate.solve_ivp` when called in
   :func:`solve_ocp` and :func:`solve_oep`.

.. py:data:: optimal.solve_ivp_options
   :type: dict
   :value: {}

   Set the value of the options keyword used by
   :func:`scipy.integrate.solve_ivp` when called in :func:`solve_ocp` and
   :func:`solve_oep`.
