"""Latent bus: activation-space fact injection in transformer language models.

A small reproducible demo showing that a single low-dimensional vector,
injected into a transformer's residual stream at answer-start, can transmit
a task-critical fact to the model — flipping an otherwise-unsolvable query
to the correct answer with zero visible tokens.

See the README for background and reproduction instructions.
"""

__version__ = "0.1.0"

from latent_bus.model_io import (
    ActivationCapture,
    load_local_model_and_tokenizer,
    select_device_and_dtype,
    set_offline_mode,
)
from latent_bus.injection import answer_start_injection
from latent_bus.run import cache_patched_payloads, extract_donor_payload

__all__ = [
    "ActivationCapture",
    "answer_start_injection",
    "cache_patched_payloads",
    "extract_donor_payload",
    "load_local_model_and_tokenizer",
    "select_device_and_dtype",
    "set_offline_mode",
]
