#!/usr/bin/env python

from distutils.core import setup

setup(name='control',
      version='0.3b',
      description='Python Control Systems Library',
      author='Richard Murray',
      author_email='murray@cds.caltech.edu',
      url='http://www.cds.caltech.edu/~murray/wiki/python-control',
      requires='scipy',
      package_dir = {'control':'src'},
      packages=['control'],
     )
