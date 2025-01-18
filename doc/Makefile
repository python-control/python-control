# Makefile for python-control Sphinx documentation
# RMM, 15 Jan 2025

FIGS = figures/classes.pdf
RST_FIGS = figures/flatsys-steering-compare.png	\
  figures/iosys-predprey-open.png				\
  figures/timeplot-servomech-combined.png			\
  figures/steering-optimal.png figures/ctrlplot-servomech.png	\
  figures/phaseplot-dampedosc-default.png			\
  figures/timeplot-mimo_step-default.png			\
  figures/freqplot-siso_bode-default.png			\
  figures/pzmap-siso_ctime-default.png				\
  figures/rlocus-siso_ctime-default.png				\
  figures/stochastic-whitenoise-response.png

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = PythonControlLibrary
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile html latexpdf doctest clean distclean

# List of the first RST figure of each type in each file that is generated
figures/flatsys-steering-compare.png: flatsys.rst
	@$(SPHINXBUILD) -M doctest "$(SOURCEDIR)" "$(BUILDDIR)"
figures/iosys-predprey-open.png: iosys.rst
	@$(SPHINXBUILD) -M doctest "$(SOURCEDIR)" "$(BUILDDIR)"
figures/timeplot-servomech-combined.png: nlsys.rst
	@$(SPHINXBUILD) -M doctest "$(SOURCEDIR)" "$(BUILDDIR)"
figures/steering-optimal.png: optimal.rst
	@$(SPHINXBUILD) -M doctest "$(SOURCEDIR)" "$(BUILDDIR)"
figures/phaseplot-dampedosc-default.png: phaseplot.rst
	@$(SPHINXBUILD) -M doctest "$(SOURCEDIR)" "$(BUILDDIR)"
figures/timeplot-mimo_step-default.png		\
  figures/freqplot-siso_bode-default.png	\
  figures/pzmap-siso_ctime-default.png		\
  figures/rlocus-siso_ctime-default.png		\
  figures/ctrlplot-servomech.png: response.rst
	@$(SPHINXBUILD) -M doctest "$(SOURCEDIR)" "$(BUILDDIR)"

figures/stochastic-whitenoise-response.png: stochastic.rst
	@$(SPHINXBUILD) -M doctest "$(SOURCEDIR)" "$(BUILDDIR)"

# Other figure rules
figure/classes.pdf: figure/classes.fig
	make -C figures classes.pdf

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
html latexpdf doctest clean: Makefile $(FIGS) $(RST_FIGS)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

distclean: clean
	/bin/rm -rf generated
