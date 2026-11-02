# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ThinkingBudgetStateHolder batch index moves."""

import pytest
import torch

from tests.v1.sample.utils import create_mock_reasoning_config
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    MoveDirectionality,
)
from vllm.v1.sample.thinking_budget_state import (
    CumulativeThinkingBudget,
    ThinkingBudgetStateHolder,
)


def _make_holder() -> ThinkingBudgetStateHolder:
    return ThinkingBudgetStateHolder(
        create_mock_reasoning_config([151667], [151668]),
        8,
        0,
        torch.device("cpu"),
        False,
    )


def test_swap_budgeted_with_unbudgeted_clears_empty_side():
    """Asymmetric SWAP must not leave the empty index sharing state."""
    h = _make_holder()
    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=[
                (0, SamplingParams(thinking_token_budget=5), None, []),
                (1, SamplingParams(), None, []),
            ],
            moved=(),
        )
    )
    assert list(h._state.keys()) == [0]
    budget_state = h._state[0]

    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=(),
            moved=[(0, 1, MoveDirectionality.SWAP)],
        )
    )
    assert list(h._state.keys()) == [1]
    assert h._state[1] is budget_state
    assert h._state[1]["thinking_token_budget"] == 5

    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=(),
            moved=[(0, 1, MoveDirectionality.SWAP)],
        )
    )
    assert list(h._state.keys()) == [0]
    assert h._state[0] is budget_state


def test_swap_exchanges_two_budgeted_states():
    h = _make_holder()
    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=[
                (0, SamplingParams(thinking_token_budget=3), None, []),
                (1, SamplingParams(thinking_token_budget=7), None, []),
            ],
            moved=(),
        )
    )
    b0 = h._state[0]["thinking_token_budget"]
    b1 = h._state[1]["thinking_token_budget"]
    h.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            added=(),
            moved=[(0, 1, MoveDirectionality.SWAP)],
        )
    )
    assert h._state[0]["thinking_token_budget"] == b1
    assert h._state[1]["thinking_token_budget"] == b0


def test_cumulative_budget_counts_all_blocks_and_forecasts_without_spending():
    budget = CumulativeThinkingBudget([1], [2], 5)
    for token in [1, 10, 11, 2, 20, 1, 12]:
        budget.accept(token)
    assert budget.remaining == 2
    assert budget.forecast([13, 14, 15]) == [(2, 2)]
    assert budget.remaining == 2  # proposals are not accepted tokens
    budget.accept(19)  # different accepted token after a rejected draft
    assert budget.forecast([20]) == [(1, 2)]
    budget.accept(20)
    budget.accept(2)
    budget.accept(1)
    assert budget.forced_token() == 2


@pytest.mark.parametrize("limit", [0, 1, 5])
def test_cumulative_budget_handles_split_boundaries(limit):
    budget = CumulativeThinkingBudget([1, 3], [2, 4], limit)
    for token in [1, 3] + [10] * limit:
        budget.accept(token)
    assert budget.forced_token() == 2
    budget.accept(2)
    assert budget.forced_token() == 4
    budget.accept(4)
    assert budget.remaining == 0 and budget.forced_token() is None
    budget.accept(1)
    assert budget.forced_token() is None
    budget.accept(3)
    assert budget.forced_token() == 2


def test_cumulative_partial_end_cannot_spend_beyond_remaining_budget():
    budget = CumulativeThinkingBudget([1], [2, 4], 1)
    budget.accept(1)
    budget.accept(2)
    # A mismatch would turn both prefix and next token into reasoning, exceeding
    # the one-token allowance. Complete the end marker instead.
    assert budget.forced_token() == 4
    budget.accept(4)
    budget.accept(1)
    budget.accept(10)
    assert budget.remaining == 0 and budget.forced_token() == 2


def test_cumulative_natural_end_preserves_unspent_allowance():
    budget = CumulativeThinkingBudget([1], [20, 2], 3, natural_end=[2])
    for token in [1, 10, 2, 1, 11, 12]:
        budget.accept(token)
    assert budget.forced_token() == 20
    budget.accept(20)
    assert budget.forced_token() == 2
    budget.accept(2)
    assert budget.remaining == 0 and not budget.in_think


def test_cumulative_speculative_masks_use_remaining_accepted_budget():
    from vllm.config import ModelConfig, ReasoningConfig

    config = ReasoningConfig(
        reasoning_start_str="<think>", reasoning_end_str="</think>"
    )
    config.initialize_token_ids(
        ModelConfig(model="Qwen/Qwen3-0.6B", max_model_len=2048)
    )
    start, end = config.reasoning_start_token_ids, config.reasoning_end_token_ids
    assert start and end and len(start) == len(end) == 1
    holder = ThinkingBudgetStateHolder(config, 2, 3, torch.device("cpu"), False)
    holder.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=(),
            moved=(),
            added=[
                (
                    i,
                    SamplingParams(
                        thinking_token_budget=5, thinking_token_budget_cumulative=True
                    ),
                    start,
                    [],
                )
                for i in range(2)
            ],
        )
    )
    accepted = [10, 11] + end + start + [12]
    drafts = [[13, 14, 15], [13, 14]]
    holder.update_state([accepted, accepted], drafts)
    logits = holder.apply_to_logits(torch.zeros(5, end[0] + 1), False, drafts)
    assert torch.nonzero(logits, as_tuple=False).tolist() == [[2, end[0]]]
    bonus = holder.apply_to_logits(torch.zeros(2, end[0] + 1), True, drafts)
    assert torch.nonzero(bonus, as_tuple=False).tolist() == [[1, end[0]]]
    # Repeating an update with no new accepted output must not spend proposals.
    holder.update_state([accepted, accepted], drafts)
    repeated = holder.apply_to_logits(torch.zeros_like(logits), False, drafts)
    assert torch.equal(logits, repeated)
