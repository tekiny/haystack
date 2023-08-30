from unittest.mock import Mock

import pytest

from haystack.preview.components.generators._helpers import enforce_token_limit


@pytest.mark.unit
def test_enforce_token_limit(caplog):
    tokenizer = Mock()
    tokenizer.encode = lambda text: text.split()
    tokenizer.decode = lambda tokens: " ".join(tokens)

    assert enforce_token_limit("This is a test prompt.", tokenizer=tokenizer, max_tokens_limit=3) == "This is a"
    assert caplog.records[0].message == (
        "The prompt has been truncated from 5 tokens to 3 tokens so that the prompt fits within the max token "
        "limit. Reduce the length of the prompt to prevent it from being cut off."
    )