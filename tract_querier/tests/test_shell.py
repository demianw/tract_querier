from .. import shell


class EmptySpatialIndexing:
    crossing_tracts_labels = {}
    crossing_labels_tracts = {}
    ending_tracts_labels = ({}, {})
    ending_labels_tracts = ({}, {})
    label_bounding_boxes = {}
    tract_bounding_boxes = {}


def test_names():
    cmd = shell.TractQuerierCmd(
        EmptySpatialIndexing(),
        initial_body="a.left = 1\na.right = 2\nb = 1\nc_left = 1\nc_right = 2"
    )
    assert sorted(cmd.names()) == [
        'a.left', 'a.right', 'b', 'c_left', 'c_right'
    ]


def test_syntax_error_is_printed(monkeypatch, capsys):
    def raise_syntax_error(*args, **kwargs):
        raise SyntaxError('invalid syntax')

    monkeypatch.setattr(shell, 'queries_preprocess', raise_syntax_error)
    cmd = shell.TractQuerierCmd(EmptySpatialIndexing())
    cmd.default('a = (')
    cmd.do_save('a')
    out, err = capsys.readouterr()
    assert out.count('invalid syntax') == 2
    assert 'Uncaught exception' not in err
