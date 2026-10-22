"""Glass-break classification with YAMNet (TensorFlow Lite), fully offline.

Works with both YAMNet TFLite variants you may download:
  * the "classification" model: fixed 15600-sample input (0.975 s), outputs 521 class scores
  * the full model: variable-length input, outputs scores, 1024-number embeddings and a spectrogram

Optionally runs your own trained head model (Phase 2) on YAMNet's embeddings,
which needs the full model because only it outputs embeddings.
"""

import csv
import logging
from dataclasses import dataclass, field

import numpy as np

from .audio import rms_dbfs

log = logging.getLogger(__name__)

YAMNET_WINDOW = 15600  # 0.975 s at 16 kHz
NUM_CLASSES = 521
EMBEDDING_SIZE = 1024


@dataclass
class Result:
    glass_score: float
    is_glass: bool
    top: list = field(default_factory=list)  # [(label, score), ...] highest first
    source: str = "yamnet"

    @property
    def label(self):
        return self.top[0][0] if self.top else "unknown"


def _interpreter_class():
    """Find a TFLite interpreter: the new ai-edge-litert package first, then older ones."""
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter
    except ImportError:
        pass
    try:
        from tensorflow.lite import Interpreter  # full TensorFlow, laptop only
        return Interpreter
    except ImportError:
        pass
    raise ImportError(
        "No TensorFlow Lite runtime found. Install one with: pip install ai-edge-litert"
    )


def load_class_names(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [row["display_name"] for row in csv.DictReader(f)]


class YamnetClassifier:
    def __init__(self, cfg):
        c = cfg["classifier"]
        self.threshold = float(c.get("glass_threshold", 0.3))
        self.class_names = load_class_names(c["class_map_path"])

        wanted = [name.lower() for name in c.get("glass_classes", [])]
        lookup = {name.lower(): i for i, name in enumerate(self.class_names)}
        self.glass_idx = [lookup[name] for name in wanted if name in lookup]
        missing = [name for name in wanted if name not in lookup]
        if missing:
            log.warning("Glass classes not in the class map, skipped: %s", missing)
        if not self.glass_idx:
            raise ValueError("None of the configured glass_classes exist in the class map")
        log.info("Glass classes: %s", [self.class_names[i] for i in self.glass_idx])

        Interpreter = _interpreter_class()
        self.interp = Interpreter(model_path=c["model_path"])
        self.interp.allocate_tensors()
        inp = self.interp.get_input_details()[0]
        self.input_index = inp["index"]
        signature = list(inp.get("shape_signature", inp["shape"]))
        self.dynamic = any(d == -1 for d in signature)
        self.input_rank = len(inp["shape"])
        self.window = YAMNET_WINDOW if self.dynamic else int(inp["shape"][-1])
        self._find_outputs()

        self.head = None
        if c.get("head_model_path"):
            if self.embeddings_index is None:
                raise ValueError(
                    "head_model_path needs the full YAMNet model (one that outputs embeddings); "
                    "this model only outputs class scores"
                )
            self.head = Interpreter(model_path=c["head_model_path"])
            self.head.allocate_tensors()
            self.head_glass_index = int(c.get("head_glass_index", 0))
            log.info("Using custom head model %s", c["head_model_path"])

        log.info(
            "YAMNet loaded: %s input, window %d samples, embeddings %s",
            "variable" if self.dynamic else "fixed",
            self.window,
            "yes" if self.embeddings_index is not None else "no",
        )

    def _find_outputs(self):
        self.scores_index = None
        self.embeddings_index = None
        for out in self.interp.get_output_details():
            last = int(out["shape"][-1]) if len(out["shape"]) else 0
            if last == NUM_CLASSES and self.scores_index is None:
                self.scores_index = out["index"]
            elif last == EMBEDDING_SIZE and self.embeddings_index is None:
                self.embeddings_index = out["index"]
        if self.scores_index is None:
            raise ValueError("Could not find a 521-class score output; is this a YAMNet model?")

    def _run(self, chunk):
        """Run YAMNet on one chunk. Returns (scores [frames, 521], embeddings [frames, 1024] or None)."""
        chunk = chunk.astype(np.float32)
        if self.dynamic:
            self.interp.resize_tensor_input(self.input_index, [len(chunk)])
            self.interp.allocate_tensors()
        elif self.input_rank == 2:
            chunk = chunk.reshape(1, -1)
        self.interp.set_tensor(self.input_index, chunk)
        self.interp.invoke()
        scores = self.interp.get_tensor(self.scores_index).reshape(-1, NUM_CLASSES)
        emb = None
        if self.embeddings_index is not None:
            emb = self.interp.get_tensor(self.embeddings_index).reshape(-1, EMBEDDING_SIZE)
        return scores, emb

    def _chunks(self, samples):
        """Fixed-input models: overlapping windows (50% hop) covering the whole clip."""
        if len(samples) <= self.window:
            yield np.pad(samples, (0, self.window - len(samples)))
            return
        hop = self.window // 2
        start = 0
        while True:
            piece = samples[start:start + self.window]
            if len(piece) < self.window:
                piece = np.pad(piece, (0, self.window - len(piece)))
            yield piece
            if start + self.window >= len(samples):
                break
            start += hop

    def classify(self, samples):
        if self.dynamic:
            padded = samples if len(samples) >= self.window else np.pad(samples, (0, self.window - len(samples)))
            all_scores, all_emb = self._run(padded)
        else:
            parts = [self._run(chunk) for chunk in self._chunks(samples)]
            all_scores = np.concatenate([s for s, _ in parts])
            embs = [e for _, e in parts if e is not None]
            all_emb = np.concatenate(embs) if embs else None

        # A glass break is short, so use the loudest frame (max), not the average.
        peak = all_scores.max(axis=0)
        top = [(self.class_names[i], float(peak[i])) for i in np.argsort(peak)[::-1][:5]]

        if self.head is not None:
            glass_score = self._head_score(all_emb)
            source = "custom-head"
        else:
            glass_score = float(peak[self.glass_idx].max())
            source = "yamnet"

        return Result(glass_score, glass_score >= self.threshold, top, source)

    def _head_score(self, embeddings):
        inp = self.head.get_input_details()[0]
        out = self.head.get_output_details()[0]
        best = 0.0
        for frame in embeddings:
            self.head.set_tensor(inp["index"], frame.reshape(inp["shape"]).astype(np.float32))
            self.head.invoke()
            probs = self.head.get_tensor(out["index"]).reshape(-1)
            best = max(best, float(probs[self.head_glass_index]))
        return best


class DummyClassifier:
    """Pipeline testing only: calls any loud clip 'glass'. Not AI, never deploy it."""

    def __init__(self, cfg):
        self.threshold = float(cfg["classifier"].get("glass_threshold", 0.3))
        log.warning("Using the DUMMY classifier: loud sounds count as glass. Testing only.")

    def classify(self, samples):
        db = rms_dbfs(samples)
        score = float(np.clip((db + 40.0) / 30.0, 0.0, 1.0))  # -40 dBFS -> 0, -10 dBFS -> 1
        return Result(score, score >= self.threshold, [("Loud sound (dummy)", score)], "dummy")


def make_classifier(cfg):
    backend = cfg["classifier"].get("backend", "yamnet")
    if backend == "dummy":
        return DummyClassifier(cfg)
    if backend == "yamnet":
        return YamnetClassifier(cfg)
    raise ValueError(f"unknown classifier backend: {backend}")
