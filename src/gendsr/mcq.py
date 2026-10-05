"""Canonical multiple-choice text shared by training and evaluation."""

from __future__ import annotations

import re
from collections.abc import Sequence


OPTION_LETTERS = ("A", "B", "C", "D")
LETTER_ONLY_OUTPUT_INSTRUCTION = "Respond with only A, B, C, or D."
LETTER_ONLY_OUTPUT_SUFFIX = f"\n\n{LETTER_ONLY_OUTPUT_INSTRUCTION}"


def _option_values(options: Sequence[str]) -> tuple[str, ...]:
    if isinstance(options, (str, bytes)) or len(options) != len(OPTION_LETTERS):
        raise ValueError("Multiple-choice records require exactly four options")
    values = tuple(str(option).strip() for option in options)
    if any(not value for value in values):
        raise ValueError("Multiple-choice options must not be empty")

    punctuated = tuple(
        re.sub(
            rf"^\s*{letter}\s*[\.\):]\s*",
            "",
            value,
            count=1,
            flags=re.IGNORECASE,
        ).strip()
        for letter, value in zip(OPTION_LETTERS, values, strict=True)
    )
    if punctuated != values:
        return punctuated

    # Only strip unpunctuated labels when all four are present. This avoids
    # treating an unlabeled value such as ``A person turns left`` as a prefix.
    plain_labelled = all(
        re.match(rf"^\s*{letter}\s+", value, flags=re.IGNORECASE)
        for letter, value in zip(OPTION_LETTERS, values, strict=True)
    )
    if plain_labelled:
        return tuple(
            re.sub(
                rf"^\s*{letter}\s+",
                "",
                value,
                count=1,
                flags=re.IGNORECASE,
            ).strip()
            for letter, value in zip(OPTION_LETTERS, values, strict=True)
        )
    return values


def format_mcq_text(question: str, options: Sequence[str]) -> str:
    """Return the exact question/options text used by DSR SFT records."""

    question = str(question).strip()
    if not question:
        raise ValueError("Multiple-choice question must not be empty")
    values = _option_values(options)
    lines = [question, "Options:"]
    lines.extend(
        f"{letter} {value}"
        for letter, value in zip(OPTION_LETTERS, values, strict=True)
    )
    return "\n".join(lines)


def build_sft_aligned_mcq_prompt(question_and_options: str) -> str:
    """Add only the generation-format constraint missing from SFT inputs."""

    text = str(question_and_options).strip()
    if not text:
        raise ValueError("SFT-aligned MCQ prompt must not be empty")
    return f"{text}{LETTER_ONLY_OUTPUT_SUFFIX}"
