# Makefile for python-control examples
# RMM, 6 Jul 2024
#
# This makefile allows cleanup and posting of Jupyter notebooks into
# Google Colab.

NOTEBOOKS = cds110-L*_*.ipynb cds112-L*_*.ipynb
GDRIVE= gdrive:projects/python-control/public/notebooks

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
post: .ipynb-clean
	rclone copy . $(GDRIVE) --include /cds110-L\*_\*.ipynb
	rclone copy . $(GDRIVE) --include /cds112-W\*_\*.ipynb
