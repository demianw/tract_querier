import inspect
from itertools import repeat
from collections import Mapping, Iterable
import io
import csv
from os import path

from six.moves import range

import nibabel

from ..tractography import (
    Tractography, tractography_to_file
)
import os


def set_dictionary_from_use_filenames_as_index(
    optional_flags,
    tractography_name, default_tractography_name,
    results, measurement_dict
):
    """
    Parse the use_file_names_as_index option and set dictionary accordingly
    """
    file_name_components_to_use = optional_flags.get(
        "--use_file_names_as_index", None
    )
    if file_name_components_to_use is not None:
        if len(file_name_components_to_use) == 0:
            results.setdefault('tract_file_path', []).append(tractography_name)
        else:
            file_name_elements = tractography_name.split(os.path.sep)
            for file_path_index_to_record in file_name_components_to_use:
                col_name = file_path_index_to_record.split(":")[0]

                element_index_from_end = (
                    -1 *
                    int(file_path_index_to_record.split(":")[1])
                )

                results.setdefault(col_name, []).append(
                    file_name_elements[element_index_from_end]
                )
    else:
        results.setdefault('tract file #', []).append(
            default_tractography_name)
    for meas_k, meas_v in measurement_dict.items():
        results.setdefault(meas_k, []).append(meas_v)
    return results

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
    def parse_optional_args(optional_args, current_dict):
        if len(optional_args) > 0:
            values_list = list()
            key = optional_args[0]
            optional_args = optional_args[1:]
            while (len(optional_args) > 0) and (not optional_args[0].startswith("--")):
                values_list.append(optional_args[0])
                optional_args = optional_args[1:]
            current_dict[key] = values_list
            return parse_optional_args(optional_args, current_dict)
        else:
            return current_dict

    def find_optional_args(args):
        """ Re-organize args to put a dictionary first as the optional arguments
        args=['tract_math', '/tmp/proj/subj/session/t1.vtp','/tmp/proj/subj/session/t2.vtp',
               '--file_names','-1','-2','-3','--another_option']
        new_args=find_optional_args(args)
        print(new_args)
        [{'--file_names': ['-1', '-2', '-3'], '--another_option': []},
             'tract_math', '/tmp/proj/subj/session/t1.vtp', '/tmp/proj/subj/session/t2.vtp']
        """
        base_args = args
        optional_args = list()
        for index in range(len(args)):
            if isinstance(args[index], str) and args[index].startswith("--"):
                base_args = args[:index]
                optional_args = args[index:]
                break
        options_dict = parse_optional_args(optional_args, dict())
        return base_args, options_dict

    def internal_decorator(func):
        def wrapper(*input_args):
            # find optional flags and put them as a dictionary
            # at the beginning of args
            args, options_dict = find_optional_args(input_args)

            total_args = len(args)
            argspec = inspect.getargspec(func)
            # Subtract 1 for implicit options_dict
            func_total_args = len(argspec.args) - 1

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
                    # Subtract 1 for implicit options_dict
                    ix = argspec.args.index('file_output') - 1

                    if ix >= len(args):
                        raise TractMathWrongArgumentsError(
                            'Wrong number of parameters for the operation:\n' +
                            str(args) + "\n" + str(ix) +
                            " != " + str(len(args))
                        )
                    file_output = args[ix]
                else:
                    file_output = args[-1]
                    args = args[:-1]

                option_and_args = (options_dict,) + args
                out = func(*option_and_args)

                process_output(out, file_output=file_output)
            elif (
                total_args >= func_total_args or
                len(args) == func_args
            ):
                if args[-1] == '-':
                    args = args[:-1]

                option_and_args = (options_dict,) + args
                process_output(func(*option_and_args))
            else:
                raise TractMathWrongArgumentsError(
                    'Wrong number of arguments for the operation'
                    + str(args) + "\n"
                )

        wrapper.help_text = help_text
        wrapper.needs_one_tract = needs_one_tract

        return wrapper

    return internal_decorator


def process_output(output, file_output=None):
    if output is None:
        return

    if file_output is not None and path.exists(file_output):
        in_key = input("Overwrite file %s (y/N)? " % file_output)
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
            f = io.StringIO()
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
                for row in zip(
                    repeat(output.keys(), len(first_value)),
                    zip(*output.values())
                )
            )
            writer.writerows(rows)
        else:
            writer.writerow(output)

        if file_output is None:
            print(f.getvalue())


class TractMathWrongArgumentsError(TypeError):
    pass
