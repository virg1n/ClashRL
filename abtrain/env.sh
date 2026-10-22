# Sourced by setup_ab.sh and run_ab.sh: every cache and temp file goes under $AB_WORK
# (the network home), never to the nearly full root disk.
export AB_WORK="${AB_WORK:-$HOME/clashrl_ab}"
export AB_FL_REPO="${AB_FL_REPO:-$AB_WORK/ext/FirstLight_CR}"
mkdir -p "$AB_WORK/tmp" "$AB_WORK/cache"
export TMPDIR="$AB_WORK/tmp"
export XDG_CACHE_HOME="$AB_WORK/cache"
export PIP_CACHE_DIR="$AB_WORK/cache/pip"
export UV_CACHE_DIR="$AB_WORK/cache/uv"
export UV_PYTHON_INSTALL_DIR="$AB_WORK/python"
export HF_HOME="$AB_WORK/cache/hf"
export TORCH_HOME="$AB_WORK/cache/torch"
export TRITON_CACHE_DIR="$AB_WORK/cache/triton"
export CUDA_CACHE_PATH="$AB_WORK/cache/nv"
export MPLCONFIGDIR="$AB_WORK/cache/mpl"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1
