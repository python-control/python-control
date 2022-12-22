""" set-pip-test-matrix.py

Create test matrix for pip wheels
"""
import json
from pathlib import Path

system_opt_blas_libs = {'ubuntu': ['OpenBLAS'],
                        'macos' : ['OpenBLAS', 'Apple']}

wheel_jobs = []
for wkey in Path("slycot-wheels").iterdir():
    wos, wpy, wbl = wkey.name.split("-")
    wheel_jobs.append({'packagekey':  wkey.name,
                        'os': wos,
                        'python': wpy,
                        'blas_lib': wbl,
                        })
    if wbl == "Generic":
        for bl in system_opt_blas_libs[wos]:
            wheel_jobs.append({ 'packagekey':  wkey.name,
                                'os': wos,
                                'python': wpy,
                                'blas_lib': bl,
                                })

matrix = { 'include': wheel_jobs }
print(json.dumps(matrix))