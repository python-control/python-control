# Makefile for python-control examples
# RMM, 6 Jul 2024
#
# This makefile allows cleanup and posting of Jupyter notebooks into
# Google Colab.
#
# Files are copied to Google Colab using rclone.  In order to copy files to
# Google Colab, you should edit the GDRIVE variable to use the name of the
# drive you have configured in rclone and the path where you want to place
# the files.  The default location is set up for the fbsbook.org@gmail.com
# Google Drive account, currently maintained by Richard Murray.

NOTEBOOKS = cds110-L*_*.ipynb cds112-L*_*.ipynb
GDRIVE= fbsbook-gdrive:python-control/public/notebooks

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
