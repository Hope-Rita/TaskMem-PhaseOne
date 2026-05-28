# Installs the exact dependency versions used during development.

set -e

PIP="${PIP:-pip}"
SUDO="${USE_SUDO:-sudo}"

$SUDO $PIP install --no-cache-dir \
    compressed-tensors==0.11.0 \
    frozendict==2.4.6 \
    lm-format-enforcer==0.11.3 \
    openai==1.99.1 \
    openai-harmony==0.0.4 \
    outlines_core==0.2.11 \
    xformers==0.0.32.post1 \
    xgrammar==0.1.25 \
    qwen-vl-utils==0.0.14 \
    tokenizers==0.22.1 \
    transformers==4.57.1 \
    uvloop==0.21.0 \
    flashinfer-python==0.2.2

if command -v nvidia-smi &> /dev/null && nvidia-smi > /dev/null 2>&1; then
    $SUDO $PIP install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 \
        torch==2.8.0+cu128 torchaudio==2.8.0+cu128 torchvision==0.23.0+cu128
    $SUDO $PIP install -v --no-build-isolation 'transformer_engine[pytorch]==2.8.0'

    # Pre-built flash-attn wheel matching cu128/torch2.8/cp310
    FLASH_ATTN_WHEEL="flash_attn-2.7.4+cu128torch2.8-cp310-cp310-linux_x86_64.whl"
    if [ ! -f "$FLASH_ATTN_WHEEL" ]; then
        wget https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.18/$FLASH_ATTN_WHEEL
    fi
    $SUDO $PIP install --no-cache-dir $FLASH_ATTN_WHEEL

    $SUDO $PIP install vllm==0.11.0 --no-deps
    $SUDO $PIP uninstall -y mbridge || true
    $SUDO $PIP install -U git+https://github.com/ISEEKYAN/mbridge.git@892727e02b1b16318732e9677751f65fe971b2bd
    $SUDO $PIP install --no-deps --no-cache-dir git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.13.1
else
    $SUDO $PIP install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0
    $SUDO $PIP install -v --no-build-isolation 'transformer_engine[pytorch]==2.8.0'
    $SUDO $PIP install vllm==0.11.0 --no-deps
fi

$PIP install "numpy<2.0.0"
$PIP install json_repair
$PIP install protobuf==3.20.2
$PIP install httpx==0.23.3