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

MAJOR = 0
MINOR = 6
MICRO = 5
ISRELEASED = True
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
PACKAGE_NAME        = 'control'
EXTRA_INFO          = dict(
    install_requires=['scipy', 'matplotlib'],
    tests_require=['scipy', 'matplotlib', 'nose']
)

VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

import os
import sys
import subprocess


from setuptools import setup, find_packages

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


# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


def get_version_info():
    # Adding the git rev number needs to be done inside write_version_py(),
    # otherwise the import of package.version messes up
    # the build under Python 3.
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists('control/version.py'):
        # must be a source distribution, use existing version file
        try:
            from control.version import git_revision as GIT_REVISION
        except ImportError:
            raise ImportError("Unable to import git_revision. Try removing "
                              "control/version.py and the build directory "
                              "before building.")
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += '.dev-' + GIT_REVISION[:7]

    return FULLVERSION, GIT_REVISION


def write_version_py(filename='control/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s

if not release:
    version = full_version
"""
    FULLVERSION, GIT_REVISION = get_version_info()

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'git_revision': GIT_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()


def setup_package():
    src_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    # Rewrite the version file everytime
    write_version_py()

    metadata = dict(
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
        classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
        platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
        install_requires=['numpy', 'scipy'],
        tests_require=['nose'],
        test_suite='nose.collector',
        packages=find_packages(
            exclude=['*.tests']
        ),
    )

    FULLVERSION, GIT_REVISION = get_version_info()
    metadata['version'] = FULLVERSION

    try:
        setup(**metadata)
    finally:
        del sys.path[0]
        os.chdir(old_path)
    return

if __name__ == '__main__':
    setup_package()
