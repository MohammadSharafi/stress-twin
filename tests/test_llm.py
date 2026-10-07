import json

import numpy as np
import pytest

from stress_twin.llm import LLMUnavailable, TokenFactory, extract_json, pick_model

AVAILABLE = ["nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B",
             "nvidia/Nemotron-3-Nano-Omni-30B-A3B", "nvidia/llama-nemotron-embed-1b-v2", "Qwen/Qwen3-235B-A22B"]


def test_pick_model_prefers_ultra_and_skips_embeddings():
    assert pick_model(AVAILABLE, ("nemotron-3-ultra", "nemotron")) == AVAILABLE[0]
    assert pick_model(AVAILABLE, ("nano-omni", "cosmos")) == AVAILABLE[2]
    assert pick_model(["nvidia/llama-nemotron-embed-1b-v2"], ("nemotron",)) is None
    assert pick_model(["nvidia/cosmos3-super-reasoner"], ("nano-omni", "cosmos3")) == "nvidia/cosmos3-super-reasoner"


def test_extract_json_tolerates_reasoning_wrappers():
    assert extract_json('<think>x</think>\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure: {"b": [1, 2]} done') == {"b": [1, 2]}


def test_from_env_requires_key(monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    with pytest.raises(LLMUnavailable):
        TokenFactory.from_env()


def test_roles_schema_vision_and_log(mock_tf, tmp_path):
    log = tmp_path / "calls.jsonl"
    tf = TokenFactory("test-key", mock_tf.base_url, log_path=log)
    models = tf.models()
    assert models["planner"].endswith("Nemotron-3-Ultra-550B-A55B")
    assert models["fast"].endswith("Nemotron-3.5-Lightning-30B-A3B")
    assert models["vision"].endswith("Nemotron-3-Nano-Omni-30B-A3B")

    schema = {"type": "object", "properties": {"scenarios": {"type": "array"}}, "required": ["scenarios"]}
    data = tf.chat_json("planner", "sys", "Propose exactly 3 new scenarios.", schema)
    assert len(data["scenarios"]) == 3
    req = mock_tf.state["requests"][-1]
    assert req["response_format"] == {"type": "json_schema", "json_schema": schema}
    assert json.dumps(schema) in req["messages"][0]["content"]

    img = np.zeros((8, 8, 3), dtype=np.uint8)
    cap = tf.chat_json("vision", "sys", "perception error 40 mm", {"type": "object"}, images=[img, img])
    assert cap["cause"] == "perception_miss"
    parts = mock_tf.state["requests"][-1]["messages"][1]["content"]
    assert [p["type"] for p in parts] == ["text", "image_url", "image_url"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")

    lines = [json.loads(l) for l in log.read_text().splitlines()]
    assert [l["role"] for l in lines] == ["planner", "vision"]
    assert all(l["ok"] and l["prompt_tokens"] is not None and l["latency_s"] >= 0 for l in lines)


def test_falls_back_to_json_object(mock_tf, tmp_path):
    mock_tf.state["reject_json_schema"] = True
    tf = TokenFactory("test-key", mock_tf.base_url, log_path=tmp_path / "c.jsonl")
    tf.client.max_retries = 0
    data = tf.chat_json("planner", "sys", "Propose exactly 2 new scenarios.", {"type": "object"})
    assert len(data["scenarios"]) == 2
    assert [c["response_format"] for c in tf.calls] == ["json_schema", "json_object"]
    assert [c["ok"] for c in tf.calls] == [False, True]
