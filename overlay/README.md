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

The image reuses vLLM's official CUDA artifacts from the base commit, pinned by
digest in `base.env`. Changed Python files are copied over the installed
package, and the bundled Rust frontend is rebuilt from this branch so request
fields have the same behavior through either frontend. The Rust build uses the
same package and `native-tls-vendored` feature as `tools/build_rust.py`.

`build-context.sh` stages the Python overlay and complete Rust build inputs.
Deleted or renamed Python files, CUDA/C++ changes, and other runtime inputs it
cannot carry still fail staging and require a full image build. Tests, docs,
Buildkite configuration and these build files do not alter the runtime overlay.

On each push to a branch that contains it,
[overlay-image.yml](../.github/workflows/overlay-image.yml) builds the image and
pushes it as `ghcr.io/richardjennings/vllm:<branch>-<commit>` (branch slashes are
replaced with hyphens).

## Moving to a newer vLLM

1. Choose a main commit that has an official image,
   `vllm/vllm-openai:cu134-nightly-<commit>`.
2. Start a new branch from that commit. Merge the pull requests that are still
   open, and bring over `overlay/` and the workflow.
3. Set `BASE_COMMIT` and `BASE_IMAGE`, with the image's digest, in `base.env`.
