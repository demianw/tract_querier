from importlib.metadata import version

import tract_querier


def test_version_is_the_distribution_version():
    assert tract_querier.__version__ == version('tract_querier')


def test_cli_version_option(script_runner):
    ret = script_runner.run(["tract_querier", "--version"])
    assert ret.success
    assert ret.stdout.strip() == tract_querier.__version__
