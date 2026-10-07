# Gates — Stress Twin build

- [x] G1 Test suite passes
  CHECK: .venv/bin/pytest -q 2>&1 | tail -3
  EXPECT: passed
  EVIDENCE: ......................                                                   [100%] | 22 passed in 23.01s

- [x] G2 Every scenario across the full parameter bounds builds and simulates (no MuJoCo errors, 200 random scenarios)
  CHECK: .venv/bin/python -m stress_twin.cli selftest --n 200 2>&1 | tail -1
  EXPECT: selftest ok
  EVIDENCE: selftest ok

- [x] G3 Policy under test is competent in its nominal domain: success >= 85% on 200 nominal scenarios
  CHECK: .venv/bin/python -m stress_twin.cli nominal --n 200 2>&1 | tail -1
  EXPECT: NOMINAL_OK
  EVIDENCE: NOMINAL_OK

- [x] G4 Policy has a real failure surface: >= 3 distinct failure causes observed in 300 uniform scenarios
  CHECK: .venv/bin/python -m stress_twin.cli surface --n 300 2>&1 | tail -1
  EXPECT: SURFACE_OK
  EVIDENCE: SURFACE_OK

- [x] G5 Episodes are deterministic: same scenario+seed gives identical result and telemetry hash (tested)
  CHECK: .venv/bin/pytest -q tests/test_episode.py 2>&1 | tail -1
  EXPECT: passed
  EVIDENCE: 4 passed in 2.26s

- [ ] G6 Boundary search beats random stress testing at equal budget, mean over 3 seeds, on BOTH: subtle failures found (stress <= 0.30, the smallest perturbations that break the policy) AND distinct failure modes (cause x primary stressor)
  AMENDED 2026-10-07 before any benchmark ran: raw failure count was dropped because a fully random 14-parameter scenario fails 200/200 times, so it rewards brute force, not insight.
  CHECK: .venv/bin/python -m stress_twin.cli benchmark --budget 240 --seeds 3 2>&1 | tail -1
  EXPECT: BENCHMARK_OK
  EVIDENCE: pending
ABANDON: G6 the distinct-modes half fails at budget 240 (15.7 vs 15.7, a tie). Subtle failures pass (27.0 vs 7.0). scripts/mode_universe.py shows a ceiling effect: both strategies find every common mode, the rest occur 1-5 times per 3000 random episodes. Budget curve, reported in full in README: 120 -> modes 11.7 vs 13.3 (random ahead), 480 -> 18.7 vs 18.3. An exploration share was tried and removed because it cut subtle failures to 16.0 without improving modes.

- [x] G7 Token Factory client works end to end against an OpenAI-compatible mock: model discovery, role routing, json_schema output, vision message, call log
  CHECK: .venv/bin/pytest -q tests/test_llm.py 2>&1 | tail -1
  EXPECT: passed
  EVIDENCE: 5 passed in 1.47s

- [x] G8 LLM strategy + analyst run a full campaign against the mock server, captions grounding-checked against telemetry
  CHECK: .venv/bin/pytest -q tests/test_llm_campaign.py 2>&1 | tail -1
  EXPECT: passed
  EVIDENCE: 1 passed in 10.02s

- [x] G9 Nebius Serverless Jobs runner: builds the documented CLI command and parses results, tested with a fake `nebius` CLI
  CHECK: .venv/bin/pytest -q tests/test_nebius_runner.py 2>&1 | tail -1
  EXPECT: passed
  EVIDENCE: 4 passed in 14.91s

- [x] G10 Offline end-to-end campaign produces a report with failure map, clusters with clips, and safety case
  CHECK FIX 2026-10-07: the first check counted matching lines (11), not distinct markers; intent unchanged.
  CHECK: .venv/bin/python -m stress_twin.cli run --budget 240 --rounds 4 --strategy boundary --out runs/demo >/dev/null 2>&1; for m in 'id="failure-map"' 'id="clusters"' 'id="safety-case"' 'data:image/gif'; do grep -q "$m" runs/demo/report.html && echo found; done | wc -l | sed 's/^ */markers=/'
  EXPECT: markers=4
  EVIDENCE: markers=4

- [x] G11 Safety envelope claim is validated on held-out scenarios: pass rate inside envelope >= 90%
  CHECK: .venv/bin/python -m stress_twin.cli validate --run runs/demo --n 150 2>&1 | tail -1
  EXPECT: ENVELOPE_OK
  EVIDENCE: ENVELOPE_OK

- [x] G12 README is honest and complete: setup, offline vs live modes, every NVIDIA/Nebius component and where it is used, Apache-2.0 LICENSE present
  CHECK FIX 2026-10-07: LICENSE starts with a blank line; read the title line by pattern instead.
  CHECK: for m in "Nemotron 3 Ultra" "Nano Omni" "Token Factory" "Serverless Jobs" "Not yet run" "## Limitations" "## Quick start" "## Live mode"; do grep -q "$m" README.md && echo found; done | wc -l | sed 's/^ */readme_markers=/'; grep -m1 -o "Apache License" LICENSE
  EXPECT: Apache License
  EVIDENCE: readme_markers=8 | Apache License

- [ ] G13 Live Token Factory campaign with Nemotron (requires NEBIUS_API_KEY)
  EVIDENCE: pending
ABANDON: G13 NEBIUS_API_KEY is not available in this environment. The planner and vision paths are implemented and exercised end to end against an OpenAI-compatible mock (G7, G8). Command to close this gate: `stress-twin run --strategy llm --analyst llm --budget 120 --rounds 4 --out runs/live`.

- [ ] G14 Live Serverless Jobs run (requires Nebius account + pushed container image)
  EVIDENCE: pending
ABANDON: G14 no Nebius account or registry credentials here. Partial evidence: worker image built with Docker (Colima VM, aarch64, mujoco 3.14.0); scripts/container_check.py ran 24 episodes inside the container: 24/24 same outcome and cause as local, perception estimates within 1.00 mm; fake-CLI round trip passes (G9).

- [x] G15 Git repository initialised and committed locally (not pushed)
  CHECK: git -C . log --oneline | head -1
  EXPECT: Stress Twin
  EVIDENCE: cda12a4 Stress Twin: adversarial simulation testing for robot policies
