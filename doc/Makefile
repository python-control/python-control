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
figures/iosys-predprey-open.png: iosys.rst
figures/timeplot-servomech-combined.png: nlsys.rst
figures/steering-optimal.png: optimal.rst
figures/phaseplot-dampedosc-default.png: phaseplot.rst
figures/timeplot-mimo_step-default.png: response.rst
figures/freqplot-siso_bode-default.png: response.rst
figures/pzmap-siso_ctime-default.png: response.rst
figures/rlocus-siso_ctime-default.png: response.rst
figures/ctrlplot-servomech.png: response.rst
figures/stochastic-whitenoise-response.png: stochastic.rst

# Other figure rules
figure/classes.pdf: figure/classes.fig
	make -C figures classes.pdf

# Rule to run `make doctest` once for all figures
.docfigs: $(RST_FIGS)
	@$(SPHINXBUILD) -M doctest "$(SOURCEDIR)" "$(BUILDDIR)"
	touch $@

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
html latexpdf doctest clean: Makefile .docfigs $(FIGS)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

distclean: clean
	/bin/rm -rf generated
