# Makefile for python-control examples
# RMM, 6 Jul 2024
#
# This makefile allows cleanup and posting of Jupyter notebooks into
# Google Colab.

NOTEBOOKS = cds110-L*_*.ipynb
FILES = satellite-dish.png disk-drive.png robotic-arm.png		\
  springmass-coupled.png pvtol.py kincar.py estimation.png ssctrl.png	\
  zn-step-response.png zn-step-table.png zn-step-improved.png		\
  pid-aw-diagram.png
GDRIVE= gdrive:python-control/examples

# Clean up notebooks to remove output
clean: .ipynb-clean
.ipynb-clean: $(NOTEBOOKS)
	@for i in $?; do \
	    echo jupyter nbconvert --clear-output clear-metadata $$i; \
	    jupyter nbconvert \
		--ClearMetadataPreprocessor.enabled=True \
		--clear-output $$i; \
	done
	touch $@

# Post Jupyter notebooks on course website
post: .ipynb-clean $(FILES)
	rclone -u copy $(NOTEBOOKS) $(FILES) $(GDRIVE)
