"""Find the token positions where we read activations.

Two positions are supported:
- "item": the last token of the item in the prompt.
- "period": the token that contains the final period of the prompt.
"""

import re

POSITIONS = ("item", "period")


def item_char_span(prompt, item_text):
    """Return (start, end) of the first whole-word match of the item.

    The match ignores case because some prompts start with the item ("Doll: toy.").
    """
    pattern = r"\b" + re.escape(item_text) + r"\b"
    match = re.search(pattern, prompt, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Item {item_text!r} not found in prompt {prompt!r}")
    return match.start(), match.end()


def period_char_index(prompt):
    if not prompt.endswith("."):
        raise ValueError(f"Prompt does not end with a period: {prompt!r}")
    return len(prompt) - 1


def target_char_index(prompt, item_text, position):
    """Return the index of one character inside the target token."""
    if position == "item":
        _start, end = item_char_span(prompt, item_text)
        return end - 1
    if position == "period":
        return period_char_index(prompt)
    raise ValueError(f"Unknown position {position!r}, use one of {POSITIONS}")


def token_index_at_char(offsets, char_index):
    """Return the index of the token whose character span contains char_index.

    offsets is a list of (start, end) pairs from a fast tokenizer.
    Special and padding tokens have (0, 0) and never match.
    """
    for token_index, (start, end) in enumerate(offsets):
        if start <= char_index < end:
            return token_index
    raise ValueError(f"No token contains character {char_index}")
