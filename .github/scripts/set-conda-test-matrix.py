"""Create test matrix for conda packages in OS/BLAS test matrix workflow."""

import json
from pathlib import Path
import re

osmap = {'linux': 'ubuntu',
         'osx': 'macos',
         'win': 'windows',
         }

combinations = {'ubuntu': ['unset', 'Generic', 'OpenBLAS', 'Intel10_64lp'],
                'macos': ['unset', 'Generic', 'OpenBLAS'],
                'windows': ['unset', 'Intel10_64lp'],
               }

conda_jobs = []
for conda_pkg_file in Path("slycot-conda-pkgs").glob("*/*.tar.bz2"):
    cos = osmap[conda_pkg_file.parent.name.split("-")[0]]
    m = re.search(r'py(\d)(\d+)_', conda_pkg_file.name)
    pymajor, pyminor = int(m[1]), int(m[2])
    cpy = f'{pymajor}.{pyminor}'
    for cbl in combinations[cos]:
        cjob = {'packagekey': f'{cos}-{cpy}',
                'os': cos,
                'python': cpy,
                'blas_lib':  cbl}
        conda_jobs.append(cjob)

matrix = { 'include': conda_jobs }
print(json.dumps(matrix))
