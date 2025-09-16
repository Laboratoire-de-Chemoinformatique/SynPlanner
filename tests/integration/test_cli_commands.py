from click.testing import CliRunner
from synplan.interfaces.cli import synplan


def test_synplan_help():
    runner = CliRunner()
    result = runner.invoke(synplan, ["--help"])
    assert result.exit_code == 0
    assert "SynPlanner command line interface." in result.output
