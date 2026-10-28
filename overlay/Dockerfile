# vLLM's official image of BASE_COMMIT, with the Python files this branch
# changes copied over the installed package. See overlay/README.md.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
