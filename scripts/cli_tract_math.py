#!/usr/bin/env python
import sys
import itertools
from argparse import ArgumentParser, FileType, REMAINDER

import traceback

from tract_querier.tract_math import operations


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
        func.help_text = help_text
        func.needs_one_tract = needs_one_tract
        return func
    return internal_decorator


def _build_arg_parser():
    usage = r"""
    usage: %(prog)s <tract1.vtk> ... <tractN.vtk> operation <operation parameter1> ... <operation parameterN> <output_tract.vtk> \
                      [ optional flags ]

    optional flags:  Always start with "--" and are considered as boolean flags

    The --use_file_names_as_index col_name1:path_dir_index col_name2:path_dir_index2 .. col_nameN:path_dir_indexN
        Example directory layout:
        =========================
        PROJ_XX/SUBJ_01/SESSION_AB/cm1.left.vtp
        PROJ_XX/SUBJ_01/SESSION_AB/cm1.right.vtp
        ||||||| ||||||| |||||||||| ||||||||||||||||||||||||
        4444444 3333333 2222222222 111111111111111111111111

        tract_math call:
        ================
        tract_math PROJ_XX/SUBJ_01/SESSION_AB/*.vtp count test.csv --use_file_names_as_index proj:4 subj:3 sess:2 tract:1

        result:
        =======
        $ cat test.csv
        sess,number of tracts,subj,proj,tract
        SESSION_AB,176,SUBJ_01,PROJ_XX,cm1.left.vtp
        SESSION_AB,88,SUBJ_01,PROJ_XX,cm1.right.vtp

    Available operations:
    """

    operations_names = operations.keys()
    operations_names = sorted(operations_names)
    for f in operations_names:
        usage += '\t%s %s\n' % (f, operations[f].help_text)

    # The first arguments, except for the last one, might be tractography files

    n_tracts = len([
        tract for tract in
        itertools.takewhile(
            lambda x: x not in operations_names,
            sys.argv[1:]
        )
    ])

    parser = ArgumentParser(usage=usage)
    parser.add_argument('tractographies', nargs=max(
        n_tracts, 1), help='tractography files', type=FileType('r'))
    parser.add_argument('operation', type=str, choices=operations_names,
                        help="operation to use")
    parser.add_argument('operation_parameters', type=str, nargs=REMAINDER,
                        help="operation parameters")

    return parser


def main():
    from tract_querier.tract_math import TractMathWrongArgumentsError

    parser = _build_arg_parser()
    args = parser.parse_args()

    # Load the global modules after the parsing of parameters
    from tract_querier.tractography import tractography_from_files

    if (args.operation in operations) and operations[args.operation].needs_one_tract:
        tractography = tractography_from_files(
            [f.name for f in args.tractographies])
    else:
        tractography = []
        try:
            for f in args.tractographies:
                # For each tractography file, pass a tuple of (tractography
                # filename, tractography instance) to tractography list.
                tractography.append((f.name, tractography_from_files(f.name)))
        except IOError as e:
            # print >>sys.stderr, "Error reading file ", f.name, "(%s)" % repr(e)
            print(sys.stderr, "Error reading file ", f.name, "(%s)" % repr(e))

    if args.operation in operations:
        try:
            operations[args.operation](
                tractography, *args.operation_parameters)
        except TractMathWrongArgumentsError as e:
            traceback.print_exc(file=sys.stdout)
            parser.error('\n\n' + str(e))
        except TypeError:
            traceback.print_exc(file=sys.stdout)
            parser.error("\n\nError: Wrong number of parameters for the operation")

    else:
        parser.error("\n\nOperation not found")


if __name__ == "__main__":
    main()
    sys.exit()
