"""Download the model files Jitterbug uses (about 1.2 MB, plus Whisper on first use).

    python scripts/download_models.py

DNSMOS is published by Microsoft in https://github.com/microsoft/DNS-Challenge
(see that repository for its licence). Whisper models are fetched automatically
by faster-whisper the first time speech-to-text runs.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    ROOT / "models" / "dnsmos" / "sig_bak_ovr.onnx": (
        "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"
    ),
}


def main() -> None:
    for path, url in FILES.items():
        if path.exists():
            print(f"already there: {path.relative_to(ROOT)}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        print(f"saved {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes, sha256 {digest}...)")


if __name__ == "__main__":
    main()
