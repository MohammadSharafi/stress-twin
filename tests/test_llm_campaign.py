import json

from stress_twin.campaign import run_campaign
from stress_twin.llm import TokenFactory
from stress_twin.llm_strategy import LLMAdversary, campaign_brief
from stress_twin.report import build_report
from stress_twin.runners import LocalRunner


def test_llm_campaign_and_grounded_analyst(mock_tf, tmp_path, monkeypatch):
    run = tmp_path / "run"
    client = TokenFactory("test-key", mock_tf.base_url, log_path=run / "llm_calls.jsonl")
    strategy = LLMAdversary(client)
    with LocalRunner(workers=2) as runner:
        hist = run_campaign(strategy, runner, budget=14, rounds=2, seed=0, out=run, log=lambda *_: None)
    (run / "planner_notes.json").write_text(json.dumps(strategy.notes()))
    (run / "campaign.json").write_text(json.dumps({"strategy": "llm"}))

    sources = {h["source"] for h in hist}
    assert "llm" in sources and "boundary-fill" in sources  # the invalid proposal was replaced
    llm_recs = [h for h in hist if h["source"] == "llm"]
    assert all(h["hypothesis"] and "confirmed" in h for h in llm_recs)
    assert any(h["confirmed"] for h in llm_recs)
    assert all(n["accepted"] == 6 for n in strategy.notes())  # 6 valid of 7 per round

    # Round 2's prompt carried round 1's verdicts back to the planner.
    planner_prompts = [r["messages"][1]["content"] for r in mock_tf.state["requests"] if "Ultra" in r["model"]]
    assert "Your hypotheses from the last round" in planner_prompts[1]
    assert "Per-factor evidence" in campaign_brief(hist)

    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    monkeypatch.setenv("NEBIUS_BASE_URL", mock_tf.base_url)
    html = build_report(run, analyst_mode="llm", log=lambda *_: None).read_text()
    caps = json.loads((run / "captions.json").read_text())
    assert caps and all(c["analyst"].endswith("Nano-Omni-30B-A3B") for c in caps)
    for c in caps:
        rec_cause = c["mode"].split("|")[0]
        assert c["grounded"] == (c["cause"] == rec_cause)
    assert 'id="adversary"' in html and "data:image/gif" in html
