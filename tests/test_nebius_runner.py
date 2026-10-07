import os
import stat
import sys
import textwrap

import numpy as np

from stress_twin import params
from stress_twin.nebius_jobs import NebiusJobRunner, parse_results
from stress_twin.runners import LocalRunner
from stress_twin.strategies import stressed_sample

FAKE_CLI = textwrap.dedent('''\
    #!{python}
    """Fake `nebius` CLI: records `ai create` jobs and serves their logs by running the worker."""
    import json, os, subprocess, sys
    state = os.environ["FAKE_NEBIUS_STATE"]
    a = sys.argv[1:]
    if a[:2] == ["ai", "create"]:
        opts = dict(zip(a[2::2], a[3::2]))
        assert opts["--type"] == "job" and opts["--container-command"] == "python"
        for flag in ("--name", "--image", "--platform", "--preset", "--timeout", "--args"):
            assert flag in opts, flag
        json.dump(opts, open(os.path.join(state, opts["--name"] + ".json"), "w"))
        print("created")
    elif a[:3] == ["ai", "job", "get-by-name"]:
        print(a[a.index("--name") + 1])
    elif a[:3] == ["ai", "job", "logs"]:
        name = a[3]
        polls = os.path.join(state, name + ".polls")
        n = int(open(polls).read()) if os.path.exists(polls) else 0
        open(polls, "w").write(str(n + 1))
        if n == 0:
            print("job is starting")  # first poll: not done yet
        else:
            opts = json.load(open(os.path.join(state, name + ".json")))
            payload = opts["--args"].split("--payload ")[1]
            out = subprocess.run([sys.executable, "-m", "stress_twin.worker", "--payload", payload, "--workers", "2"],
                                 capture_output=True, text=True, check=True).stdout
            print(out)
    else:
        sys.exit("unsupported: " + " ".join(a))
''')


def _scenarios(n):
    rng = np.random.default_rng(11)
    return [stressed_sample(rng) for _ in range(n)]


def test_command_matches_documented_cli():
    r = NebiusJobRunner(image="cr.example/stress-twin:1", dry_run=False)
    cmd = r.create_command("job-1", "PAYLOAD")
    assert cmd[:5] == ["nebius", "ai", "create", "--type", "job"]
    opts = dict(zip(cmd[3::2], cmd[4::2]))
    assert opts["--image"] == "cr.example/stress-twin:1"
    assert opts["--container-command"] == "python"
    assert opts["--args"] == "-m stress_twin.worker --payload PAYLOAD"


def test_parse_results_waits_for_done_marker():
    assert parse_results("STRESS_TWIN_RESULT {\"i\": 0, \"result\": {\"x\": 1}}", 1) is None
    assert parse_results("STRESS_TWIN_RESULT {\"i\": 0, \"result\": {\"x\": 1}}\nSTRESS_TWIN_DONE 1", 1) == [{"x": 1}]


def test_jobs_roundtrip_with_fake_cli(tmp_path, monkeypatch):
    cli = tmp_path / "nebius"
    cli.write_text(FAKE_CLI.format(python=sys.executable))
    cli.chmod(cli.stat().st_mode | stat.S_IEXEC)
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setenv("FAKE_NEBIUS_STATE", str(state))
    scen = _scenarios(7)
    runner = NebiusJobRunner(image="cr.example/stress-twin:1", batch_size=4, cli=str(cli), poll_s=0.05, log=lambda *_: None)
    remote = runner.run(scen)
    assert len(runner.submitted) == 2 and len(list(state.glob("*.json"))) == 2
    with LocalRunner(workers=2) as local:
        expected = local.run(scen)
    assert remote == expected  # deterministic episodes: cloud and local agree exactly


def test_dry_run_executes_worker_locally():
    scen = _scenarios(3)
    runner = NebiusJobRunner(image="", batch_size=2, dry_run=True, log=lambda *_: None)
    res = runner.run(scen)
    assert [r["scenario"] for r in res] == [params.clip(s) for s in scen]
