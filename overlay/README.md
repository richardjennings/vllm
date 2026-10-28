# DeepSeek V4.1 on RTX PRO 6000

This branch is vLLM main at the commit in [base.env](base.env) with four open
upstream pull requests merged. Together they serve DeepSeek-V4.1-Flash on four
RTX PRO 6000 Blackwell GPUs (SM120), with its Engram tables read from disk:

| PR | Change |
|---|---|
| [#60762](https://github.com/vllm-project/vllm/pull/60762) | 64-token kernel blocks for the SM120 sparse attention backends |
| [#57292](https://github.com/vllm-project/vllm/pull/57292) | Fail at startup on SM120 when FlashInfer lacks runtime page sizes |
| [#63473](https://github.com/vllm-project/vllm/pull/63473) | Refuse the `flashinfer_cutlass` MoE kernel on SM120, which corrupts output |
| [#62238](https://github.com/vllm-project/vllm/pull/62238) | Engram `disk_offload`, merged with main's peer Engram tables |

## Image

The pull requests change Python files only. The image is vLLM's official image
of the base commit, pinned by digest in `base.env`, with the Python files this
branch changes copied over the installed package. `build-context.sh` stages
those files. It fails for any change a file copy cannot carry, such as a
deleted or renamed file, or a change outside `vllm/*.py` other than tests, docs
and these build files. Such a change needs a full image build.

On each push to a branch that contains it,
[overlay-image.yml](../.github/workflows/overlay-image.yml) builds the image and
pushes it as `ghcr.io/richardjennings/vllm:<branch>-<commit>`.

## Moving to a newer vLLM

1. Choose a main commit that has an official image,
   `vllm/vllm-openai:cu134-nightly-<commit>`.
2. Start a new branch from that commit. Merge the pull requests that are still
   open, and bring over `overlay/` and the workflow.
3. Set `BASE_COMMIT` and `BASE_IMAGE`, with the image's digest, in `base.env`.
