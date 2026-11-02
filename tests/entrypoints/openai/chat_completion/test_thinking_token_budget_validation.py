# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.exceptions import VLLMValidationError


@pytest.mark.parametrize("raw_value", [-2, 0.6, 10.5])
def test_chat_completion_request_rejects_invalid_thinking_token_budget(raw_value):
    with pytest.raises(VLLMValidationError, match="thinking_token_budget"):
        ChatCompletionRequest.model_validate(
            {
                "model": "qwen",
                "messages": [{"role": "user", "content": "hello"}],
                "thinking_token_budget": raw_value,
            }
        )


def test_chat_completion_request_accepts_valid_thinking_token_budget():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "thinking_token_budget": 10,
        }
    )
    assert request.thinking_token_budget == 10


def test_chat_completion_request_accepts_minus_one_as_unlimited():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "thinking_token_budget": -1,
        }
    )
    assert request.thinking_token_budget is None


@pytest.mark.parametrize("budget", [None, -1])
def test_cumulative_thinking_requires_a_finite_budget(budget):
    request = ChatCompletionRequest.model_validate(
        {
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            "thinking_token_budget": budget,
            "thinking_token_budget_cumulative": True,
        }
    )
    with pytest.raises(ValueError, match="requires thinking_token_budget"):
        request.to_sampling_params(100, {})


@pytest.mark.parametrize("raw_value", [0.6, 3.14, -2])
def test_completion_request_rejects_invalid_thinking_token_budget(raw_value):
    with pytest.raises(VLLMValidationError, match="thinking_token_budget"):
        CompletionRequest.model_validate(
            {
                "model": "qwen",
                "prompt": "hello",
                "thinking_token_budget": raw_value,
            }
        )


def test_completion_request_accepts_valid_thinking_token_budget():
    request = CompletionRequest.model_validate(
        {
            "model": "qwen",
            "prompt": "hello",
            "thinking_token_budget": 5,
        }
    )
    assert request.thinking_token_budget == 5


def test_completion_request_accepts_minus_one_as_unlimited():
    request = CompletionRequest.model_validate(
        {
            "model": "qwen",
            "prompt": "hello",
            "thinking_token_budget": -1,
        }
    )
    assert request.thinking_token_budget is None
