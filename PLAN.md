# Stress Twin — build plan

Adversarial simulation testing for robot manipulation policies. Given a policy and a
scene, search the scenario space for failures, explain them, and produce a safety case.

## Assumptions (stated up front)
- Simulator: MuJoCo (pip-installable, deterministic, renders headless on macOS/Linux).
- Robot: Cartesian gantry with a force-limited parallel gripper (real robot class, no IK needed).
- Policy under test: learned CNN perception (trained only on nominal scenes) + scripted grasp
  controller. Realistic failure surface: perception shift, slip, geometry, disturbance.
- No NEBIUS_API_KEY in this environment. Every Nebius/NVIDIA call is implemented against the
  documented API and tested against a local OpenAI-compatible mock. Offline fallbacks are
  labelled "offline" everywhere they appear. Live runs are gated on the key.
- Serverless Jobs integration uses the documented `nebius ai create --type job` CLI; results
  return via job logs, clips are re-rendered locally from the deterministic seed.

## Module contracts
- params.py: PARAMS (ordered specs), Scenario = dict[name -> physical value] + seed.
  sample_uniform(rng), sample_nominal(rng), to_unit(s)/from_unit(u), clip(s).
- scene.py: build_xml(scenario) -> str.
- episode.py: run_episode(scenario, policy, record=False) -> EpisodeResult (JSON-able:
  success, cause, telemetry summary, optional frames). Deterministic in (scenario, seed).
- diagnose.py: diagnose(telemetry) -> cause in CAUSES.
- strategies.py: Strategy.propose(history, n, rng) -> list[Scenario] (+hypothesis text).
  random | boundary (offline active learning) | llm (Nemotron via Token Factory).
- llm.py: TokenFactory client: discover models, route roles (planner/fast/vision),
  chat_json(schema), call log JSONL.
- analyst.py: caption(failure) -> {cause, explanation, confidence, grounded: bool}.
- analysis.py: clusters, factor effects, safety envelope (decision tree) + validation.
- runners.py: LocalRunner (multiprocessing), NebiusJobRunner (CLI, dry-run capable).
- campaign.py: rounds loop, writes runs/<id>/{scenarios,results,llm_calls}.jsonl + report.html.
- report.py: self-contained HTML (failure map, progress, clusters + GIFs, safety case).
- cli.py: `stress-twin train|run|benchmark|validate|report`.

## Status log
- 2026-10-07 plan written; MuJoCo headless render verified (3.15.0).
