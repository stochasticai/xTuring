"""Generate a short dummy WAV tone for testing."""

from __future__ import annotations

import argparse
import math
import wave
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a short dummy WAV tone")
    parser.add_argument(
        "--output",
        default="/tmp/dummy_tone.wav",
        help="Output WAV path.",
    )
    parser.add_argument(
        "--seconds", type=float, default=1.0, help="Duration in seconds"
    )
    parser.add_argument("--freq", type=float, default=440.0, help="Tone frequency (Hz)")
    parser.add_argument("--rate", type=int, default=16000, help="Sample rate")
    parser.add_argument("--amp", type=float, default=0.2, help="Amplitude (0-1)")

    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    n_samples = int(args.seconds * args.rate)
    with wave.open(str(output), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(args.rate)

        for i in range(n_samples):
            sample = args.amp * math.sin(2 * math.pi * args.freq * (i / args.rate))
            value = int(max(-1.0, min(1.0, sample)) * 32767)
            wf.writeframesraw(value.to_bytes(2, byteorder="little", signed=True))

    print(str(output))


if __name__ == "__main__":
    main()
