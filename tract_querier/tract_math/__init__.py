from inspect import getmembers

from decorator import tract_math_operation, TractMathWrongArgumentsError

tract_math_operations = dict()

import operations as _operations

for f in getmembers(_operations):
    if hasattr(f[1], 'help_text'):
        tract_math_operations[f[0]] = f[1]


import tract_stats as _tract_stats

for f in getmembers(_tract_stats):
    if hasattr(f[1], 'help_text'):
        tract_math_operations[f[0]] = f[1]


import tract_projections as _tract_projections

for f in getmembers(_tract_projections):
    if hasattr(f[1], 'help_text'):
        tract_math_operations[f[0]] = f[1]
