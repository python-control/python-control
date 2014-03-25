#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup

setup(name = 'control',
      version = '0.6e',
      description = 'Python Control Systems Library',
      author = 'Richard Murray',
      author_email = 'murray@cds.caltech.edu',
      url = 'http://python-control.sourceforge.net',
      install_requires = ['scipy', 'matplotlib'],
      tests_require = ['scipy', 'matplotlib', 'nose'],
      packages = ['control'],
      test_suite='nose.collector'
     )
