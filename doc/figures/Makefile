# Makefile- rules to create figures
# RMM, 26 Dec 2024

# List of figures that need to be created (first figure generated is OK)
FIGS = classes.pdf

# Location of the control package
SRCDIR = ../..

all: $(FIGS)

clean:
	/bin/rm -f $(FIGS)

classes.pdf: classes.fig
	fig2dev -Lpdf $< $@
