from subprocess import check_output
import os

def main():
    cmd = 'git describe --always --long'
    output = check_output(cmd.split()).decode('utf-8').strip().split('-')
    if len(output) == 3:
        version, build, commit = output
    else:
        raise Exception("Could not git describe, (got %s)" % output)

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

    # Write files for conda version number
    SRC_DIR = os.environ.get('SRC_DIR', '.')
    conda_version_path = os.path.join(SRC_DIR, '__conda_version__.txt')
    print("Writing %s" % conda_version_path)
    with open(conda_version_path, 'w') as conda_version:
        conda_version.write(version)

    conda_buildnum_path = os.path.join(SRC_DIR, '__conda_buildnum__.txt')
    print("Writing %s" % conda_buildnum_path)

    with open(conda_buildnum_path, 'w') as conda_buildnum:
        conda_buildnum.write(build)


if __name__ == '__main__':
    main()
