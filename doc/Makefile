# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = PythonControlLibrary
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Rules to create figures
FIGS = classes.pdf timeplot-mimo_step-default.png \
  freqplot-siso_bode-default.png rlocus-siso_ctime-default.png \
  phaseplot-dampedosc-default.png ctrlplot-servomech.png
classes.pdf: classes.fig
	fig2dev -Lpdf $< $@

timeplot-mimo_step-default.png: ../control/tests/timeplot_test.py
	PYTHONPATH=.. python $<

freqplot-siso_bode-default.png: ../control/tests/freqplot_test.py
	PYTHONPATH=.. python $<

rlocus-siso_ctime-default.png: ../control/tests/rlocus_test.py
	PYTHONPATH=.. python $<

phaseplot-dampedosc-default.png: ../control/tests/phaseplot_test.py
	PYTHONPATH=.. python $<

ctrlplot-servomech.png: ../control/tests/ctrlplot_test.py
	PYTHONPATH=.. python $<

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
html pdf clean doctest: Makefile $(FIGS)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
