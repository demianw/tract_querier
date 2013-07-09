import inspect
from itertools import izip, repeat
from collections import Mapping, Iterable
import StringIO
import csv
from os import path

import nibabel

from ..tractography import (
    Tractography, tractography_to_file
)

__all__ = ['tract_math_operation']


def tract_math_operation(help_text, needs_one_tract=True):
    '''
    Decorator to identify tract_math functionalities the name of the
    function will be automatically incorporated to the tract_math options

    Parameters
    ----------
    help_text: help for the operation
    needs_one_tract: tells the script if all the input tractographies should
                      be unified as one or left as a tractography list
    '''

    def internal_decorator(func):
        def wrapper(*args):
            total_args = len(args)
            argspec = inspect.getargspec(func)
            func_total_args = len(argspec.args)

            if argspec.varargs:
                func_total_args += 1

            func_args = func_total_args

            if argspec.defaults:
                func_args = func_total_args - len(argspec.defaults)

            has_file_output = 'file_output' in argspec.args

            if (
                (total_args > func_total_args and args[-1] != '-') or
                has_file_output
            ):
                if has_file_output:
                    ix = argspec.args.index('file_output')

                    if ix >= len(args):
                        raise TractMathWrongArgumentsError(
                            'Wrong number of parameters for the operation'
                        )
                    file_output = args[ix]
                else:
                    file_output = args[-1]
                    args = args[:-1]

                out = func(*args)

                process_output(out, file_output=file_output)
            elif (
                total_args >= func_total_args or
                len(args) == func_args
            ):
                if args[-1] == '-':
                    args = args[:-1]
                process_output(func(*args))
            else:
                raise TractMathWrongArgumentsError(
                    'Wrong number of arguments for the operation'
                )

        wrapper.help_text = help_text
        wrapper.needs_one_tract = needs_one_tract

        return wrapper

    return internal_decorator


def process_output(output, file_output=None):
    if output is None:
        return

    if file_output is not None and path.exists(file_output):
        in_key = raw_input("Overwrite file %s (y/N)? " % file_output)
        if in_key.lower().strip() != 'y':
            return

    if isinstance(output, Tractography):
        if file_output is not None:
            tractography_to_file(file_output, output)
        else:
            raise TractMathWrongArgumentsError(
                'This operation needs a tractography file output'
            )
    elif isinstance(output, nibabel.spatialimages.SpatialImage):
        nibabel.save(output, file_output)
    elif isinstance(output, Mapping):
        if file_output is None:
            f = StringIO.StringIO()
            dialect = 'excel-tab'
        else:
            if path.splitext(file_output)[-1] == '.txt':
                dialect = 'excel-tab'
            else:
                dialect = 'excel'
            f = open(file_output, 'w')
        writer = csv.DictWriter(f, output.keys(), dialect=dialect)
        writer.writeheader()

        first_value = output.values()[0]
        if (
            not isinstance(first_value, str) and
            isinstance(first_value, Iterable)
        ):
            rows = (
                dict(zip(*row))
                for row in izip(
                    repeat(output.keys(), len(first_value)),
                    izip(*output.values())
                )
            )
            writer.writerows(rows)
        else:
            writer.writerow(output)

        if file_output is None:
            print f.getvalue()


class TractMathWrongArgumentsError(TypeError):
    pass
