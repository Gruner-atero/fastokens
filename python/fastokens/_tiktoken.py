"""Acceleration for tiktoken-based slow ``transformers`` tokenizers.

Some models ship a slow (Python-backend) tokenizer that wraps a
``tiktoken.Encoding`` instead of a ``tokenizer.json`` -- Moonshot's Kimi
``TikTokenTokenizer`` (Kimi K2, K2.5 and K3) is the main one. ``patch_transformers``
only swaps fast-tokenizer backends, so these kept their original encoder: its
``_encode_text_piece`` slices the text at 400k characters, then walks every
character in Python to split runs of more than 25k whitespace or non-whitespace
characters before calling tiktoken. The walk dominates rendering a long prompt
(about 80 ms for a 240k-token chat).

This module routes those methods through ``Tokenizer.from_tiktoken`` and
``encode_segments``, reproducing both splits with a regex so the token ids stay
identical: the splits move token boundaries, so they cannot be skipped.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from typing import Any

# Mirrors of the constants in Kimi's ``TikTokenTokenizer._encode_text_piece``.
TIKTOKEN_MAX_ENCODE_CHARS = 400_000
MAX_NO_WHITESPACES_CHARS = 25_000

_LONG_RUN = re.compile(
    rf"\s{{{MAX_NO_WHITESPACES_CHARS + 1},}}|\S{{{MAX_NO_WHITESPACES_CHARS + 1},}}"
)


def split_like_tiktoken_text_piece(text: str) -> Iterator[str]:
    """Yield the substrings Kimi's ``_encode_text_piece`` hands to tiktoken.

    The text is cut every ``TIKTOKEN_MAX_ENCODE_CHARS`` characters, then each
    chunk is cut so no piece holds more than ``MAX_NO_WHITESPACES_CHARS``
    consecutive whitespace or non-whitespace characters. Only runs longer than
    that limit are cut, so a regex over the long runs replaces the per-character
    walk; ``\\s`` matches exactly the characters for which ``str.isspace()`` is
    true.
    """
    for i in range(0, len(text), TIKTOKEN_MAX_ENCODE_CHARS):
        chunk = text[i : i + TIKTOKEN_MAX_ENCODE_CHARS]
        start = 0
        for run in _LONG_RUN.finditer(chunk):
            cut = run.start() + MAX_NO_WHITESPACES_CHARS
            while cut < run.end():
                yield chunk[start:cut]
                start = cut
                cut += MAX_NO_WHITESPACES_CHARS
        yield chunk[start:]


def _is_tiktoken_text_piece_tokenizer(tokenizer: Any) -> bool:
    model = getattr(tokenizer, "model", None)
    return (
        callable(getattr(tokenizer, "_encode_text_piece", None))
        and type(model).__module__.startswith("tiktoken")
        and bool(getattr(tokenizer, "vocab_file", None))
        and bool(getattr(type(tokenizer), "pat_str", None))
    )


def accelerate_tiktoken_tokenizer(tokenizer: Any) -> bool:
    """Route a tiktoken text-piece tokenizer's encoding through fastokens.

    Replaces the instance's ``_encode_text_piece`` and, when present,
    ``_encode_chat_segments`` (a list of objects with ``text`` and
    ``allow_special``), so ``encode`` and ``apply_chat_template`` keep their
    behavior and token ids. ``allow_special`` maps to tiktoken's
    ``allowed_special="all"`` versus ``disallowed_special=()``: special-token
    text is a control token only in segments that allow it.

    Returns whether the tokenizer was accelerated; a tokenizer without this
    interface, or whose vocabulary fastokens cannot load, is left unchanged.
    """
    if getattr(tokenizer, "_fastokens_tiktoken", None) is not None:
        return True
    if not _is_tiktoken_text_piece_tokenizer(tokenizer):
        return False

    from fastokens._native import Tokenizer

    try:
        fast = Tokenizer.from_tiktoken(
            tokenizer.vocab_file,
            pattern=type(tokenizer).pat_str,
            special_tokens=dict(tokenizer.model._special_tokens),
        )
    except Exception as e:  # noqa: BLE001 - keep the original encoder on any load failure
        print(f"[fastokens] could not load {tokenizer.vocab_file} ({e!r}); keeping tiktoken.")
        return False

    def encode_pieces(pieces: Iterable[tuple[str, bool]]) -> list[int]:
        return list(fast.encode_segments(list(pieces)).ids)

    def encode_text_piece(text: str, allow_special_tokens: bool = True) -> list[int]:
        return encode_pieces(
            (piece, allow_special_tokens) for piece in split_like_tiktoken_text_piece(text)
        )

    def encode_chat_segments(segments) -> list[int]:
        return encode_pieces(
            (piece, bool(segment.allow_special))
            for segment in segments
            for piece in split_like_tiktoken_text_piece(segment.text)
        )

    tokenizer._fastokens_tiktoken = fast
    tokenizer._encode_text_piece = encode_text_piece
    if callable(getattr(tokenizer, "_encode_chat_segments", None)):
        tokenizer._encode_chat_segments = encode_chat_segments
    return True
