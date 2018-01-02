# make_version.py - generate version information
#
# Author: Clancy Rowley
# Date: 2 Apr 2015
# Modified: Richard M. Murray, 28 Dec 2017
#
# This script is used to create the version information for the python-
# control package.  The version information is now generated directly from
# tags in the git repository.  Now, *before* running setup.py, one runs
#
#   python make_version.py
#
# and this generates a file with the version information.  This is copied
# from binstar (https://github.com/Binstar/binstar) and seems to work well.
#
# The original version of this script also created version information for
# conda, but this stopped working when conda v3 was released.  Instead, we
# now use jinja templates in conda-recipe to create the conda information.
# The current version information is used in setup.py, control/__init__.py,
# and doc/conf.py (for sphinx).

from subprocess import check_output
import os

def main():
    cmd = 'git describe --always --long'
    # describe --long usually outputs "tag-numberofcommits-commitname"
    output = check_output(cmd.split()).decode('utf-8').strip().rsplit('-',2)
    if len(output) == 3:
        version, build, commit = output
    else:
        # If the clone is shallow, describe's output won't have tag and
        # number of commits.  This is a particular issue on Travis-CI,
        # which by default clones with a depth of 50.
        # This behaviour isn't well documented in git-describe docs,
        # but see, e.g., https://stackoverflow.com/a/36389573/1008142
        # and https://github.com/travis-ci/travis-ci/issues/3412
        version = 'unknown'
        build = 'unknown'
        # we don't ever expect just one dash from describe --long, but
        # just in case:
        commit = '-'.join(output)

    print("Version: %s" % version)
    print("Build: %s" % build)
    print("Commit: %s\n" % commit)

    filename = "control/_version.py"
    print("Writing %s" % filename)
    with open(filename, 'w') as fd:
        if build == '0':
            fd.write('__version__ = "%s"\n' % (version))
        else:
            fd.write('__version__ = "%s.post%s"\n' % (version, build))
        fd.write('__commit__ = "%s"\n' % (commit))

if __name__ == '__main__':
    main()
