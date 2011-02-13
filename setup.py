#!/usr/bin/env python

from distutils.core import setup

setup(name='control',
      version='0.4a',
      description='Python Control Systems Library',
      author='Richard Murray',
      author_email='murray@cds.caltech.edu',
      url='http://python-control.sourceforge.net',
      requires='scipy',
      package_dir = {'control':'src'},
      packages=['control'],
     )
