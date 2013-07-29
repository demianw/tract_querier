from inspect import getmembers

from decorator import tract_math_operation, TractMathWrongArgumentsError

import operations as _operations

operations = dict((
    (f[0], f[1]) for f in getmembers(_operations) if hasattr(f[1], 'help_text')
))
