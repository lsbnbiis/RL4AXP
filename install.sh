#!/usr/bin/env bash
# RL4AXP — AMP Design System: Environment Setup
#
# Hardware: NVIDIA RTX 6000 Ada (49 GB VRAM)
# CUDA Driver: 580.x  •  CUDA Toolkit: 12.0+ (cu130 wheel)
# Python: 3.12
#
# Root cause of the TF/PyTorch coexistence issue:
#   PyTorch loads C extensions lazily (torch._C._nn, optimizer internals).
#   TensorFlow loads its own CUDA/cuDNN shared libs that conflict with
#   not-yet-loaded PyTorch extensions at the C symbol table level.
#
# Fix (encoded in gpu_setup.py):
#   Pre-exercise ALL PyTorch subsystems (nn, optim, CUDA) before TF is
#   ever imported. gpu_setup.py MUST be the first import in every entry point.
#
# Install order matters:
#   1. torch (CUDA) first — locks PyTorch C extensions
#   2. transformers / tokenizers — depend on torch, load before TF
#   3. tensorflow + tf_keras last — TF CUDA libs load after PyTorch is settled

set -euo pipefail

VENV="${VENV:-/home/cylin/.venv}"
PY="${VENV}/bin/python"
PIP="${VENV}/bin/pip"

echo "=== RL4AXP installation ==="
echo "    venv : ${VENV}"
echo "    GPU  : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'not found')"
echo ""

# ── Create venv ────────────────────────────────────────────────────────────
if [ ! -f "${PY}" ]; then
    echo "[1/5] Creating venv at ${VENV} ..."
    python3.12 -m venv "${VENV}"
else
    echo "[1/5] venv already exists at ${VENV}"
fi

# ── Upgrade pip ────────────────────────────────────────────────────────────
echo "[2/5] Upgrading pip ..."
"${PIP}" install --upgrade pip --quiet

# ── PyTorch CUDA — MUST be first ──────────────────────────────────────────
echo "[3/5] Installing PyTorch 2.12.0 (cu130) ..."
"${PIP}" install \
    torch==2.12.0 \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --quiet

# Install HuggingFace immediately after torch (before TF)
"${PIP}" install transformers==5.10.2 tokenizers==0.22.2 --quiet

# ── TensorFlow + legacy Keras ─────────────────────────────────────────────
echo "[4/5] Installing TensorFlow 2.21.0 + tf_keras ..."
"${PIP}" install tensorflow==2.21.0 tf_keras==2.21.0 --quiet

# ── All remaining packages ─────────────────────────────────────────────────
echo "[5/5] Installing remaining packages ..."
"${PIP}" install \
    streamlit==1.58.0 \
    dash==4.2.0 \
    dash-bootstrap-components==2.0.4 \
    dash-ag-grid==31.3.1 \
    numpy==2.4.4 \
    pandas==3.0.2 \
    scipy==1.17.1 \
    "scikit-learn==1.8.0" \
    matplotlib==3.10.9 \
    plotly==6.8.0 \
    h5py==3.14.0 \
    joblib==1.5.3 \
    gensim==4.4.0 \
    biopython==1.87 \
    tqdm==4.67.3 \
    pyarrow==24.0.0 \
    --quiet

# ── Create required directories ────────────────────────────────────────────
mkdir -p peptide_optimization/logs

# ── Validate ──────────────────────────────────────────────────────────────
echo ""
echo "=== Validation ==="
"${PY}" - <<'EOF'
import faulthandler; faulthandler.enable()
import gpu_setup          # MUST be first — pre-loads all PyTorch C extensions
import torch as T
import torch.nn as nn, torch.optim as optim
print(f"  torch:  {T.__version__}  |  CUDA: {T.cuda.is_available()}  |  Device: {T.cuda.get_device_name(0) if T.cuda.is_available() else 'cpu'}")

# Verify TF/PyTorch coexistence is not broken
net = nn.Linear(8, 8).to("cuda:0" if T.cuda.is_available() else "cpu")
optim.AdamW(net.parameters(), lr=1e-3)
print("  PyTorch CUDA nn+optim: OK")

import tensorflow as tf
print(f"  tensorflow: {tf.__version__}  |  GPU visible to TF: {tf.config.list_physical_devices('GPU')}")
print("  TF/PyTorch coexistence: OK")
print("\n  ✓ Installation validated successfully")
EOF

echo ""
echo "=== Quick system test ==="
echo "  Run:  ${PY} test_system.py"
echo "  Dash dashboard:      ${PY} dash_app.py          → http://127.0.0.1:8050"
echo "  Streamlit dashboard: ${VENV}/bin/streamlit run streamlit_app.py"
echo "  Training:            ${PY} run_train.py"
