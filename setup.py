from setuptools import setup, find_packages

ver = {}
try:
    with open('control/_version.py') as fd:
        exec(fd.read(), ver)
    version = ver.get('__version__', 'dev')
except IOError:
    version = 'dev'

with open('README.rst') as fp:
    long_description = fp.read()

CLASSIFIERS = """
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: BSD License
Programming Language :: Python :: 2
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3
Programming Language :: Python :: 3.3
Programming Language :: Python :: 3.4
Programming Language :: Python :: 3.5
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

setup(
    name='control',
    version=version,
    author='Richard Murray',
    author_email='murray@cds.caltech.edu',
    url='http://python-control.sourceforge.net',
    description='Python control systems library',
    long_description=long_description,
    packages=find_packages(),
    classifiers=[f for f in CLASSIFIERS.split('\n') if f],
    install_requires=['numpy',
                      'scipy',
                      'matplotlib'],
    tests_require=['scipy',
                   'matplotlib',
                   'nose'],
    test_suite = 'nose.collector',
)
