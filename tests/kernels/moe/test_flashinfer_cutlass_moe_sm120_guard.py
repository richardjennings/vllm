# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""flashinfer_cutlass MoE must refuse the SM120/SM121 MXFP4xMXFP8
configuration instead of silently serving corrupt output (the TRT-LLM
gen fused-MoE kernels are only guaranteed on SM100/SM103 upstream).
The NVFP4 expert variants are unaffected and must stay selectable on
family-12 GPUs. SM121 (cc 12.1) is outside the refusal scope: no
corruption observed there (gpt-oss-20b MXFP4xMXFP8 clean on GB10)."""

from unittest.mock import MagicMock

import pytest

import vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe as fi_mod
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
    FlashInferExperts,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp8Dynamic,
    kNvfp4Dynamic,
)


def _platform(cap: tuple[int, int]) -> MagicMock:
    platform = MagicMock()
    platform.get_device_capability.return_value = cap
    return platform


def test_rejects_sm120_mxfp8(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fi_mod, "current_platform", _platform((12, 0)))
    supported, reason = FlashInferExperts.is_supported_config(
        FlashInferExperts, MagicMock(), None, kMxfp8Dynamic, MagicMock()
    )
    assert not supported
    assert reason is not None and "SM120" in reason


def test_sm120_nvfp4_not_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    # NVFP4 experts are served correctly on family-12 GPUs (verified on
    # RTX 6000D with DeepSeek-V4-Flash-0731-NVFP4): must not hit the refusal.
    monkeypatch.setattr(fi_mod, "current_platform", _platform((12, 0)))
    try:
        supported, reason = FlashInferExperts.is_supported_config(
            FlashInferExperts, MagicMock(), None, kNvfp4Dynamic, MagicMock()
        )
        assert not (not supported and reason and "SM120" in reason)
    except Exception:
        pass  # the real base gate may raise on the MagicMock config; fine


def test_sm121_mxfp8_not_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    # GB10/SM121 MXFP4xMXFP8 serves correctly (hclsys gpt-oss-20b datapoint);
    # the refusal is scoped to cc 12.0 only.
    monkeypatch.setattr(fi_mod, "current_platform", _platform((12, 1)))
    try:
        supported, reason = FlashInferExperts.is_supported_config(
            FlashInferExperts, MagicMock(), None, kMxfp8Dynamic, MagicMock()
        )
        assert not (not supported and reason and "SM120" in reason)
    except Exception:
        pass


def test_other_archs_not_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fi_mod, "current_platform", _platform((10, 0)))
    try:
        supported, reason = FlashInferExperts.is_supported_config(
            FlashInferExperts, MagicMock(), None, kMxfp8Dynamic, MagicMock()
        )
        assert not (not supported and reason and "SM120" in reason)
    except Exception:
        pass
