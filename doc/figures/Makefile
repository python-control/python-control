# Makefile- rules to create figures
# RMM, 26 Dec 2024

# List of figures that need to be created (first figure generated is OK)
FIGS = classes.pdf timeplot-mimo_step-default.png \
  freqplot-siso_bode-default.png rlocus-siso_ctime-default.png \
  phaseplot-dampedosc-default.png ctrlplot-servomech.png

# Location of the control package
SRCDIR = ../..

all: $(FIGS)

clean:
	/bin/rm -f $(FIGS)

classes.pdf: classes.fig
	fig2dev -Lpdf $< $@

timeplot-mimo_step-default.png: $(SRCDIR)/control/tests/timeplot_test.py
	PYTHONPATH=$(SRCDIR) python $<

freqplot-siso_bode-default.png: $(SRCDIR)/control/tests/freqplot_test.py
	PYTHONPATH=$(SRCDIR) python $<

rlocus-siso_ctime-default.png: $(SRCDIR)/control/tests/rlocus_test.py
	PYTHONPATH=$(SRCDIR) python $<

phaseplot-dampedosc-default.png: $(SRCDIR)/control/tests/phaseplot_test.py
	PYTHONPATH=$(SRCDIR) python $<

ctrlplot-servomech.png: $(SRCDIR)/control/tests/ctrlplot_test.py
	PYTHONPATH=$(SRCDIR) python $<
