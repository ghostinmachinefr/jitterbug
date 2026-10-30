"""Speech quality without a reference recording, using Microsoft's DNSMOS.

DNSMOS predicts the 1-5 listening scores of ITU-T P.835 from audio alone:

* SIG  - how clean the speech itself sounds (distortion, dropouts)
* BAK  - how quiet the background is
* OVRL - overall quality, the number people usually call "MOS"

The model and the calibration below follow the official script in
https://github.com/microsoft/DNS-Challenge/blob/master/DNSMOS/dnsmos_local.py
Run `python scripts/download_models.py` once to fetch the model file.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from jitterbug.audio import SAMPLE_RATE

INPUT_SECONDS = 9.01
DEFAULT_MODEL = Path(__file__).resolve().parents[3] / "models" / "dnsmos" / "sig_bak_ovr.onnx"

# Calibration polynomials from the official script (non-personalised model).
_P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
_P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
_P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])


@dataclass
class QualityScore:
    sig: float
    bak: float
    ovrl: float

    def to_dict(self) -> dict:
        return {k: round(v, 3) for k, v in asdict(self).items()}


class DNSMOS:
    """Lazy-loaded DNSMOS scorer. Thread-safe for inference."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = Path(
            model_path or os.environ.get("JITTERBUG_DNSMOS_MODEL", DEFAULT_MODEL)
        )
        self._session = None

    @property
    def available(self) -> bool:
        return self.model_path.exists()

    def _sess(self):
        if self._session is None:
            if not self.available:
                raise FileNotFoundError(
                    f"DNSMOS model not found at {self.model_path}. "
                    "Run: python scripts/download_models.py"
                )
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            self._session = ort.InferenceSession(
                str(self.model_path), opts, providers=["CPUExecutionProvider"]
            )
        return self._session

    def score(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> QualityScore:
        if sr != SAMPLE_RATE:
            raise ValueError("DNSMOS expects 16 kHz audio")
        audio = np.asarray(audio, dtype=np.float32)
        if len(audio) == 0 or not np.any(audio):
            return QualityScore(1.0, 1.0, 1.0)  # silence: nothing intelligible arrived
        seg_len = int(INPUT_SECONDS * sr)
        while len(audio) < seg_len:  # short clips are looped, as in the official script
            audio = np.concatenate([audio, audio])
        n_hops = int(np.floor(len(audio) / sr) - INPUT_SECONDS) + 1
        sigs, baks, ovrs = [], [], []
        sess = self._sess()
        for i in range(n_hops):
            seg = audio[i * sr : i * sr + seg_len]
            if len(seg) < seg_len:
                continue
            raw = sess.run(None, {"input_1": seg[np.newaxis, :]})[0][0]
            sigs.append(_P_SIG(raw[0]))
            baks.append(_P_BAK(raw[1]))
            ovrs.append(_P_OVR(raw[2]))
        return QualityScore(float(np.mean(sigs)), float(np.mean(baks)), float(np.mean(ovrs)))
