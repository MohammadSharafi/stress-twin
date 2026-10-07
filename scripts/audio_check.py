"""Check the pro video's audio mix: voice louder than music, no clipping, sane peak level.

Reads <mix>.wav plus the stems written next to it (<mix>-voice.wav, <mix>-music.wav)."""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def rms_db(x, gate_db=-50.0):
    x = x.mean(axis=1) if x.ndim == 2 else x
    win = 4800
    n = len(x) // win
    blocks = np.sqrt(np.mean(x[: n * win].reshape(n, win) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(blocks)
    active = db[db > gate_db]
    return float(10 * np.log10(np.mean(10 ** (active / 10)))) if len(active) else -120.0


def main(mix_path):
    mix_p = Path(mix_path)
    mix, sr = sf.read(mix_p)
    voice, _ = sf.read(mix_p.with_name(mix_p.stem + "-voice.wav"))
    music, _ = sf.read(mix_p.with_name(mix_p.stem + "-music.wav"))
    peak = float(np.max(np.abs(mix)))
    clipped = int(np.sum(np.abs(mix) >= 0.999))
    v, m = rms_db(voice), rms_db(music, gate_db=-70)
    print(f"sample rate {sr}, duration {len(mix) / sr:.1f}s, peak {20 * np.log10(peak):.1f} dBFS, "
          f"clipped samples {clipped}, voice {v:.1f} dB, music {m:.1f} dB, voice-over-music {v - m:.1f} dB")
    ok = sr == 48000 and clipped == 0 and peak < 0.99 and (v - m) >= 10
    print("AUDIO_OK" if ok else "AUDIO_BAD")


if __name__ == "__main__":
    main(sys.argv[1])
