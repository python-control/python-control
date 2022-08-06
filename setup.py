from setuptools import setup, find_packages

ver = {}
try:
    with open('control/_version.py') as fd:
        exec(fd.read(), ver)
    version = ver.get('__version__', 'dev')
except IOError:
    version = '0.0.0dev'

with open('README.rst') as fp:
    long_description = fp.read()

CLASSIFIERS = """
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: BSD License
Programming Language :: Python :: 3
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
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
    author='Python Control Developers',
    author_email='python-control-developers@lists.sourceforge.net',
    url='http://python-control.org',
    project_urls={
        'Source': 'https://github.com/python-control/python-control',
    },
    description='Python Control Systems Library',
    long_description=long_description,
    packages=find_packages(exclude=['benchmarks']),
    classifiers=[f for f in CLASSIFIERS.split('\n') if f],
    install_requires=['numpy',
                      'scipy>=1.3',
                      'matplotlib'],
    extras_require={
       'test': ['pytest', 'pytest-timeout'],
       'slycot': [ 'slycot>=0.4.0' ],
       'cvxopt': [ 'cvxopt>=1.2.0' ]
    }
)
