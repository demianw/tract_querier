#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

import pytest

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


def _synthetic_tractography(shift=0.):
    import numpy
    from tract_querier.tractography import Tractography

    rng = numpy.random.RandomState(0)
    x = numpy.linspace(1, 21, 21)
    tracts = [
        numpy.c_[x, numpy.full_like(x, 5 + y), numpy.full_like(x, 5 + z)] +
        shift + rng.normal(scale=.1, size=(len(x), 3))
        for y in range(3) for z in range(3)
    ]
    data = {'fa': [numpy.linspace(.1, .9, len(t))[:, None] for t in tracts]}
    return Tractography(
        tracts, data, affine=numpy.eye(4), image_dims=numpy.array([30, 30, 30])
    )


def _synthetic_tract_file(tmp_path, name, shift=0.):
    from tract_querier.tractography import tractography_to_file

    file_name = str(tmp_path / name)
    tractography_to_file(file_name, _synthetic_tractography(shift))
    return file_name


def _synthetic_image(tmp_path, data):
    import numpy
    import nibabel

    file_name = str(tmp_path / 'image.nii.gz')
    nibabel.save(nibabel.Nifti1Image(data, numpy.eye(4)), file_name)
    return file_name


def test_tract_kappa(tmp_path, capsys):
    from tract_querier.tract_math import operations

    other = _synthetic_tract_file(tmp_path, 'other.trk', shift=3.)
    operations['tract_kappa'](_synthetic_tractography(), '1', other)
    rows = capsys.readouterr().out.strip().splitlines()
    assert rows[0] == 'tract file\tkappa value'
    assert rows[1].startswith(other)


def test_tract_tract_confidence(tmp_path):
    from tract_querier.tractography import tractography_from_files
    from tract_querier.tract_math import operations

    output = str(tmp_path / 'out.trk')
    operations['tract_tract_confidence'](
        _synthetic_tractography(), '3', 'fa', output
    )
    assert 'tprob' in tractography_from_files(output).tracts_data()


@pytest.mark.parametrize(
    'operation,parameters', [
        ('tract_prototype_median', []),
        ('tract_prototype_mean', ['0']),
    ]
)
def test_tract_prototype(tmp_path, operation, parameters):
    from tract_querier.tractography import tractography_from_files
    from tract_querier.tract_math import operations

    output = str(tmp_path / 'out.trk')
    operations[operation](_synthetic_tractography(), *parameters, output)
    assert len(tractography_from_files(output).tracts()) == 1


def test_tract_bhattacharyya_coefficient(tmp_path, capsys):
    from tract_querier.tract_math import operations

    other1 = _synthetic_tract_file(tmp_path, 'other1.trk', shift=1.)
    other2 = _synthetic_tract_file(tmp_path, 'other2.trk')
    operations['tract_bhattacharyya_coefficient'](
        _synthetic_tractography(), '1', other1, other2, '-'
    )
    rows = capsys.readouterr().out.strip().splitlines()
    assert len(rows) == 3
    assert rows[1].startswith(other1)
    assert rows[2].startswith(other2)
    assert [float(v) for v in rows[2].split('\t')[1:]] == pytest.approx([1.] * 3)


def test_tract_flip_endpoints_in_label(tmp_path):
    import numpy
    from tract_querier.tractography import tractography_from_files
    from tract_querier.tract_math import operations

    labels = numpy.zeros((30, 30, 30), dtype=numpy.int16)
    labels[18:] = 1
    image = _synthetic_image(tmp_path, labels)
    output = str(tmp_path / 'out.trk')
    operations['tract_flip_endpoints_in_label'](
        _synthetic_tractography(), image, '1', output
    )
    tractography = tractography_from_files(output)
    for tract, fa in zip(tractography.tracts(), tractography.tracts_data()['fa']):
        assert tract[0, 0] > tract[-1, 0]
        assert fa[0] > fa[-1]


def test_tract_generate_population_probability_map_single(tmp_path):
    pytest.importorskip('scipy')
    import numpy
    import nibabel
    from tract_querier.tract_math import operations

    image = _synthetic_image(tmp_path, numpy.zeros((30, 30, 30)))
    output = str(tmp_path / 'out.nii.gz')
    operations['tract_generate_population_probability_map'](
        [('t.trk', _synthetic_tractography())], image, '0', output
    )
    assert nibabel.load(output).get_fdata().max() == 1


def test_tract_deform_rejects_non_deformation(tmp_path):
    pytest.importorskip('scipy')
    import numpy
    from tract_querier.tract_math import operations

    image = _synthetic_image(tmp_path, numpy.zeros((30, 30, 30, 2)))
    with pytest.raises(ValueError, match='not a deformation field'):
        operations['tract_deform'](
            _synthetic_tractography(), image, str(tmp_path / 'out.trk')
        )


def test_scalar_mean_std(capsys):
    from tract_querier.tract_math import operations

    operations['scalar_mean_std'](_synthetic_tractography(), 'fa')
    assert 'mean fa' in capsys.readouterr().out


def test_tract_map_image_4d(tmp_path):
    pytest.importorskip('scipy')
    import numpy
    from tract_querier.tractography import tractography_from_files
    from tract_querier.tract_math import operations

    data = numpy.ones((30, 30, 30, 2))
    data[..., 1] = 2
    image = _synthetic_image(tmp_path, data)
    tractography = _synthetic_tractography()
    operations['tract_map_image'](
        tractography, image, 'value', str(tmp_path / 'out.trk')
    )
    for i in (0, 1):
        result = tractography_from_files(str(tmp_path / ('out_%04d.trk' % i)))
        values = result.tracts_data()['value']
        assert [len(v) for v in values] == [len(t) for t in tractography.tracts()]
        numpy.testing.assert_allclose(numpy.vstack(values), i + 1, rtol=1e-5)
