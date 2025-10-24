"""Entity recognition helpers backed by spaCy with graceful fallbacks."""

from __future__ import annotations

import logging
import re
from typing import Iterable, Sequence

from .features import EntityRecognizer, EntitySpan

try:  # pragma: no cover - optional dependency
    import spacy
    from spacy.language import Language
    from spacy.pipeline import EntityRuler
except ModuleNotFoundError:  # pragma: no cover - spaCy not installed
    spacy = None  # type: ignore[assignment]
    Language = None  # type: ignore[assignment]
    EntityRuler = None  # type: ignore[assignment]

logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.INFO)

_DEFAULT_PATTERNS: Sequence[dict[str, object]] = (
    {"label": "PERSON", "pattern": [{"IS_TITLE": True}, {"IS_TITLE": True}]},
    {"label": "ORG", "pattern": [{"IS_UPPER": True, "LENGTH": {">=": 2}}]},
    {"label": "PRODUCT", "pattern": [{"TEXT": {"REGEX": r"(?i)[a-z]+[- ]\d{2,}"}}]},
    {
        "label": "EVENT",
        "pattern": [{"TEXT": {"REGEX": r"(?i)summit|conference|world cup"}}],
    },
    {"label": "GPE", "pattern": [{"IS_TITLE": True}, {"IS_TITLE": True, "OP": "?"}]},
)

_UPPERCASE_TOKEN = re.compile(r"\b[A-Z]{2,}\b")
_ALPHANUM_TOKEN = re.compile(r"\b[A-Za-z0-9]{3,}\b")


class SpaCyEntityRecognizer:
    """Adapter that exposes spaCy Named Entities via the EntityRecognizer protocol."""

    def __init__(self, nlp: Language) -> None:
        self._nlp = nlp

    def __call__(self, text: str) -> Iterable[EntitySpan]:
        doc = self._nlp(text)
        for ent in doc.ents:
            yield EntitySpan(label=ent.label_, text=ent.text)


class RegexEntityRecognizer:
    """Lightweight fallback when spaCy models are unavailable."""

    def __call__(self, text: str) -> Iterable[EntitySpan]:
        seen: set[tuple[str, str]] = set()
        for match in _UPPERCASE_TOKEN.finditer(text):
            token = match.group()
            if (token, "ORG") not in seen:
                seen.add((token, "ORG"))
                yield EntitySpan(label="ORG", text=token)
        for match in _ALPHANUM_TOKEN.finditer(text):
            token = match.group()
            if token.isdigit():
                continue
            if (token, "MISC") not in seen:
                seen.add((token, "MISC"))
                yield EntitySpan(label="MISC", text=token)


def build_spacy_entity_recognizer(model: str | None = None) -> SpaCyEntityRecognizer:
    """Return a spaCy-backed recognizer, falling back to a patterned blank pipeline."""
    if spacy is None:  # pragma: no cover - handled by fallback
        raise ModuleNotFoundError("spaCy is not installed")

    nlp: Language
    if model:
        try:
            nlp = spacy.load(model)  # type: ignore[call-arg]
        except OSError as exc:  # pragma: no cover - model missing at runtime
            logger.warning("Could not load spaCy model '%s': %s", model, exc)
            nlp = spacy.blank("en")
    else:
        nlp = spacy.blank("en")

    if "ner" not in nlp.pipe_names:
        ruler: EntityRuler = nlp.add_pipe("entity_ruler")  # type: ignore[assignment]
        ruler.add_patterns(list(_DEFAULT_PATTERNS))
    return SpaCyEntityRecognizer(nlp)


def build_default_entity_recognizer() -> EntityRecognizer:
    """Return an entity recognizer that prefers spaCy but degrades gracefully."""
    if spacy is not None:
        try:
            recognizer = build_spacy_entity_recognizer()
            return recognizer
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning(
                "Falling back to regex recognizer (%s): %s",
                exc.__class__.__name__,
                exc,
            )
    return RegexEntityRecognizer()
