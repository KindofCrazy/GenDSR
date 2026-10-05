"""Chat-template-derived assistant label spans."""

from __future__ import annotations

from typing import List, Sequence

import torch


def _token_ids(tokenizer, text: str) -> List[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    values = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]


def _assistant_delimiters(tokenizer):
    probe = [{"role": "user", "content": "x"}]
    without_prompt = tokenizer.apply_chat_template(
        probe,
        tokenize=False,
        add_generation_prompt=False,
    )
    with_prompt = tokenizer.apply_chat_template(
        probe,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not with_prompt.startswith(without_prompt):
        raise ValueError("Chat template generation prompt is not a suffix")
    assistant_prefix = _token_ids(tokenizer, with_prompt[len(without_prompt) :])
    if not assistant_prefix:
        raise ValueError("Chat template produced an empty assistant prefix")
    eos_token = getattr(tokenizer, "eos_token", None)
    if not eos_token:
        raise ValueError("Tokenizer must define eos_token for assistant span masking")
    assistant_suffix = _token_ids(tokenizer, eos_token + "\n") or _token_ids(tokenizer, eos_token)
    return assistant_prefix, assistant_suffix


def _find_subsequence(values: Sequence[int], pattern: Sequence[int], start: int) -> int:
    stop = len(values) - len(pattern) + 1
    for index in range(start, max(start, stop)):
        if list(values[index : index + len(pattern)]) == list(pattern):
            return index
    return -1


def labels_from_assistant_spans(input_ids: torch.Tensor, tokenizer, ignore_index: int) -> torch.Tensor:
    """Mask all tokens except assistant content/end markers using the active template."""

    prefix, suffix = _assistant_delimiters(tokenizer)
    labels = torch.full_like(input_ids, ignore_index)
    for row_index, row in enumerate(input_ids.tolist()):
        cursor = 0
        while cursor < len(row):
            prefix_start = _find_subsequence(row, prefix, cursor)
            if prefix_start < 0:
                break
            answer_start = prefix_start + len(prefix)
            suffix_start = _find_subsequence(row, suffix, answer_start)
            if suffix_start < 0:
                break
            answer_end = suffix_start + len(suffix)
            labels[row_index, answer_start:answer_end] = input_ids[
                row_index, answer_start:answer_end
            ]
            cursor = answer_end
    return labels
