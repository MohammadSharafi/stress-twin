"""Synthesize narration sentences with Kokoro (Apache-2.0 neural TTS).

Runs in a separate environment that has kokoro-onnx installed:
    KOKORO_DIR=<dir with kokoro-v1.0.onnx and voices-v1.0.bin> <that env>/bin/python scripts/tts_kokoro.py job.json
job.json = {"voice": "af_heart", "speed": 1.0, "out_dir": "...", "sentences": {"id": "text", ...}}
espeak-ng must be installed (macOS: brew install espeak-ng); its library and data are found via
PHONEMIZER_ESPEAK_LIBRARY / PHONEMIZER_ESPEAK_DATA_PATH or Homebrew's default prefix.
"""

import json
import os
import sys
from pathlib import Path

import soundfile as sf
from kokoro_onnx import EspeakConfig, Kokoro


def main(job_path: str) -> None:
    job = json.loads(Path(job_path).read_text())
    kdir = Path(os.environ["KOKORO_DIR"])
    lib = os.environ.get("PHONEMIZER_ESPEAK_LIBRARY", "/opt/homebrew/opt/espeak-ng/lib/libespeak-ng.dylib")
    data = os.environ.get("PHONEMIZER_ESPEAK_DATA_PATH", "/opt/homebrew/opt/espeak-ng/share/espeak-ng-data")
    k = Kokoro(str(kdir / "kokoro-v1.0.onnx"), str(kdir / "voices-v1.0.bin"),
               espeak_config=EspeakConfig(lib_path=lib, data_path=data))
    out = Path(job["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    durations = {}
    for sid, text in job["sentences"].items():
        samples, sr = k.create(text, voice=job.get("voice", "af_heart"), speed=job.get("speed", 1.0), lang="en-us")
        sf.write(out / f"{sid}.wav", samples, sr)
        durations[sid] = len(samples) / sr
    (out / "durations.json").write_text(json.dumps(durations, indent=2))
    print(f"synthesized {len(durations)} sentences")


if __name__ == "__main__":
    main(sys.argv[1])
