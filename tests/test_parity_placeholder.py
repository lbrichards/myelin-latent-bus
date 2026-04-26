"""The parity-padding helper produces text with the requested token count."""

from latent_bus.prepare import _build_placeholder


class WhitespaceTokenizer:
    """A minimal tokenizer stand-in: one token per whitespace-split word.

    Good enough to exercise the loop in `_build_placeholder`, which only
    calls `tokenizer.encode(text)` and inspects `len(...)`.
    """
    def encode(self, text):
        return text.split()


def test_placeholder_matches_short_target():
    tok = WhitespaceTokenizer()
    text = _build_placeholder(tok, target_token_count=5)
    assert len(tok.encode(text)) == 5


def test_placeholder_matches_longer_target():
    tok = WhitespaceTokenizer()
    text = _build_placeholder(tok, target_token_count=20)
    assert len(tok.encode(text)) == 20


def test_placeholder_shrinks_when_target_is_small():
    tok = WhitespaceTokenizer()
    # Base sentence has 6 words; ask for 3 — we should shrink.
    text = _build_placeholder(tok, target_token_count=3)
    assert len(tok.encode(text)) <= 3
