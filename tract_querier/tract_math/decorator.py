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
            argspec = inspect.getargspec(func)
            func_args = argspec.args

            has_file_output = 'file_output' in func_args

            if argspec.defaults is None:
                defaults = tuple()
            else:
                defaults = argspec.defaults

            kwargs = {}

            if argspec.keywords is not None:
                kwargs['file_output'] = args[-1]
                args = args[:-1]
            elif argspec.varargs is not None:
                file_output = None
            elif has_file_output:
                if (
                    func_args[-1] != 'file_output' and
                    argspec.keywords is None
                ):
                    raise TractMathWrongArgumentsError('Output file reserved parameter file_output must be the last one')

                func_args = func_args[:-1]
                if args[-1] == '-':
                    file_output = None
                else:
                    file_output = args[-1]

                args = args[:-1]
                defaults = defaults[:-1]
            else:
                file_output = None
                if args[-1] == '-':
                    args = args[:-1]

            if argspec.varargs is None:
                needed_defaults = len(func_args) - len(args)
                if needed_defaults > len(defaults):
                        raise TractMathWrongArgumentsError('Wrong number of arguments')
                elif needed_defaults == -1:
                    file_output = args[-1]
                    args = args[:-1]

                if needed_defaults > 0:
                    args += defaults[-needed_defaults:]

            out = func(*args)
            process_output(out, file_output=file_output)

        wrapper.help_text = help_text
        wrapper.needs_one_tract = needs_one_tract
        wrapper.original_function = func

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

        if hasattr(writer, 'writeheader'):
            writer.writeheader()
        else:
            header = dict(zip(writer.fieldnames, writer.fieldnames))
            writer.writerow(header)

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
