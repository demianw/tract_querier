#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

from tract_querier.tests import datasets


def test_help_option(script_runner):
    ret = script_runner.run(["tract_math", "--help"])
    assert ret.success


def test_tract_math_count(script_runner):
    test_data = datasets.TestDataSet()
    tract_file = test_data.files["tract_file"]
    ret = script_runner.run(["tract_math", tract_file, "count"])
    assert re.search('[^0-9]6783[^0-9]', ret.stdout) is not None
    assert ret.success


def test_compute_all_measures_std():
    import numpy
    from tract_querier.tractography import Tractography
    from tract_querier.tract_math.tensor_operations import compute_all_measures

    lengths = numpy.array([10., 20., 40., 70.])
    values = numpy.array([.2, .4, .5, .9])
    tracts = [
        numpy.c_[numpy.linspace(0, length, 11), numpy.zeros((11, 2))]
        for length in lengths
    ]
    data = {'fa': [numpy.full((11, 1), value) for value in values]}
    tractography = Tractography(tracts, data)

    measures = compute_all_measures(
        tractography,
        ['length mean (mm)', 'length std (mm)',
         'per tract distance weighted mean %s',
         'per tract distance weighted std %s'],
        scalars=['fa']
    )

    numpy.testing.assert_allclose(measures['length std (mm)'], lengths.std())
    mean = numpy.average(values, weights=lengths)
    numpy.testing.assert_allclose(
        measures['per tract distance weighted mean fa'], mean
    )
    numpy.testing.assert_allclose(
        measures['per tract distance weighted std fa'],
        numpy.sqrt(numpy.average((values - mean) ** 2, weights=lengths))
    )

    tractography = Tractography(tracts[:1] * 4, data)
    measures = compute_all_measures(
        tractography,
        ['per tract distance weighted mean %s',
         'per tract distance weighted std %s'],
        scalars=['fa']
    )
    numpy.testing.assert_allclose(
        measures['per tract distance weighted std fa'], values.std()
    )
