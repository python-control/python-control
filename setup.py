#!/usr/bin/env python

from distutils.core import setup

setup(name = 'control',
      version = '0.4c',
      description = 'Python Control Systems Library',
      author = 'Richard Murray',
      author_email = 'murray@cds.caltech.edu',
      url = 'http://python-control.sourceforge.net',
      requires = ['scipy', 'matplotlib'],
      package_dir = {'control' : 'src'},
      packages = ['control'],
     )
