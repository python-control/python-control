#!/usr/bin/env python
descr = """Python Control Systems Library

The Python Control Systems Library, python-control,
is a python module that implements basic operations
for analysis and design of feedback control systems.

Features:
Linear input/output systems in state space and frequency domain
Block diagram algebra: serial, parallel and feedback interconnections
Time response: initial, step, impulse
Frequency response: Bode and Nyquist plots
Control analysis: stability, reachability, observability, stability margins
Control design: eigenvalue placement, linear quadratic regulator
Estimator design: linear quadratic estimator (Kalman filter)

"""

DISTNAME            = 'control'
DESCRIPTION         = 'Python control systems library'
LONG_DESCRIPTION    = descr
AUTHOR              = 'Richard Murray'
AUTHOR_EMAIL        = 'murray@cds.caltech.edu'
MAINTAINER          = AUTHOR
MAINTAINER_EMAIL    = AUTHOR_EMAIL
URL                 = 'http://python-control.sourceforge.net'
LICENSE             = 'BSD'
DOWNLOAD_URL        = URL
VERSION             = '0.6e'
PACKAGE_NAME        = 'control'
EXTRA_INFO          = dict(
    install_requires=['scipy', 'matplotlib'],
    tests_require=['scipy', 'matplotlib', 'nose']
)

CLASSIFIERS = """\
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: BSD License
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

import os
import sys
import subprocess

import setuptools
from numpy.distutils.core import setup

def configuration(parent_package='', top_path=None, package_name=DISTNAME):
    if os.path.exists('MANIFEST'): os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)

    # Avoid non-useful msg: "Ignoring attempt to set 'name' (from ... "
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage(PACKAGE_NAME)
    return config

def get_version():
    """Obtain the version number"""
    import imp
    mod = imp.load_source('version', os.path.join(PACKAGE_NAME, 'version.py'))
    return mod.__version__

# Documentation building command
try:
    from sphinx.setup_command import BuildDoc as SphinxBuildDoc
    class BuildDoc(SphinxBuildDoc):
        """Run in-place build before Sphinx doc build"""
        def run(self):
            ret = subprocess.call([sys.executable, sys.argv[0], 'build_ext', '-i'])
            if ret != 0:
                raise RuntimeError("Building Scipy failed!")
            SphinxBuildDoc.run(self)
    cmdclass = {'build_sphinx': BuildDoc}
except ImportError:
    cmdclass = {}

def write_version_py(filename='control/version.py'):
    template = """# THIS FILE IS GENERATED FROM THE CONTROL SETUP.PY
version='%s'
"""
    cwd = os.path.dirname(__file__)
    with open(os.path.join(cwd, filename), 'w') as vfile:
        vfile.write(template % VERSION)

# Call the setup function
if __name__ == "__main__":
    write_version_py()

    setup(configuration=configuration,
          name=DISTNAME,
          author=AUTHOR,
          author_email=AUTHOR_EMAIL,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          download_url=DOWNLOAD_URL,
          long_description=LONG_DESCRIPTION,
          include_package_data=True,
          test_suite="nose.collector",
          cmdclass=cmdclass,
          version=VERSION,
          classifiers=[a for a in CLASSIFIERS.split('\n') if a],
          **EXTRA_INFO)
