"""A local stand-in for the Nebius Token Factory OpenAI-compatible API, for tests.

It serves /v1/models and /v1/chat/completions, records every request, and answers like
the real roles would: the planner returns hypothesis-driven scenarios (wrapped in a think
block and code fence, as reasoning models often do), the vision model returns a caption.
"""

from __future__ import annotations

import json
import re
import socket
import threading
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

MODELS = [
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B",
    "nvidia/Nemotron-3.5-Lightning-30B-A3B",
    "nvidia/Nemotron-3-Nano-Omni-30B-A3B",
    "nvidia/llama-nemotron-embed-1b-v2",
    "Qwen/Qwen3-235B-A22B",
]

PLANS = [
    ({"light": 0.35}, "perception_miss", "Below the trained light range the CNN loses the cube's contrast."),
    ({"obj_mass": 0.9}, "slip", "Grip force cannot hold 0.9 kg at nominal friction."),
    ({"disturbance": 0.035}, "disturbance", "A 35 mm bump after perception leaves the estimate stale."),
    ({"hue_shift": 0.42}, "perception_miss", "A cube far from red is outside the training distribution."),
    ({"distractors": 2, "distractor_hue_gap": 0.03}, "perception_miss", "Near-red distractors confuse the CNN."),
    ({"obj_size": 0.036, "obj_yaw": 40}, "collision", "A large rotated cube exceeds the gripper opening."),
    ({"not_a_param": 3.0}, "slip", "Invalid proposal that must be rejected."),
]


def make_app(state: dict) -> FastAPI:
    app = FastAPI()

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": m, "object": "model", "created": 0, "owned_by": "nebius"} for m in MODELS]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        state["requests"].append(body)
        fmt = (body.get("response_format") or {}).get("type")
        if state.get("reject_json_schema") and fmt == "json_schema":
            return JSONResponse({"error": {"message": "json_schema not supported for this model"}}, status_code=400)
        model = body["model"]
        user = body["messages"][-1]["content"]
        text = user if isinstance(user, str) else " ".join(p.get("text", "") for p in user if p.get("type") == "text")
        if "Ultra" in model:
            n = int(re.search(r"Propose exactly (\d+)", text).group(1))
            items = [{"hypothesis": h, "expected_cause": c, "changes": ch} for ch, c, h in (PLANS * 10)[:n]]
            content = "<think>planning</think>\n```json\n" + json.dumps({"analysis": "Mock planner round.", "scenarios": items}) + "\n```"
        elif "Omni" in model:
            m = re.search(r"perception error (\d+) mm", text)
            err = int(m.group(1)) if m else 0
            cause = "perception_miss" if err > 15 else "slip"
            content = json.dumps({"cause": cause, "explanation": f"Mock caption: perception error {err} mm.",
                                  "visual_evidence": "red X far from the green circle", "confidence": 0.8})
        else:
            content = json.dumps({"ok": True})
        return {"id": "mock", "object": "chat.completion", "created": int(time.time()), "model": model,
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": len(text) // 4, "completion_tokens": len(content) // 4, "total_tokens": 0}}

    return app


class MockServer:
    def __init__(self) -> None:
        self.state = {"requests": []}
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.server = uvicorn.Server(uvicorn.Config(make_app(self.state), host="127.0.0.1", port=self.port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1/"

    def __enter__(self):
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                break
            time.sleep(0.05)
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=5)
