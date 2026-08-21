"""
GPU context initialisation — must be imported BEFORE any TF/Keras module.

Root cause of segfault:
  PyTorch loads some C extensions lazily (e.g. torch._C._nn, optimizer internals).
  TensorFlow/tf_keras loads its own shared libraries (cuDNN, cuBLAS, TCMalloc) that
  conflict at the C symbol level with those not-yet-loaded PyTorch extensions.
  If any PyTorch C extension is loaded AFTER TF, the loader crashes.

Fix: eagerly exercise ALL needed PyTorch subsystems BEFORE importing TF.
  - Create a dummy network, optimizer, and CUDA op → forces all .so files to load
  - Then TF can load its own libraries without conflicting with already-loaded PyTorch libs

TF is then restricted to CPU via tf.config.set_visible_devices([], 'GPU') so that
cuBLAS/cuDNN are used exclusively by PyTorch at runtime.
"""

import os

# ── Suppress TF log spam ───────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"]      = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"]     = "0"
os.environ["KERAS_BACKEND"]             = "tensorflow"
os.environ["TF_USE_LEGACY_KERAS"]       = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ── Step 1: Eagerly load ALL PyTorch subsystems before TF touches anything ─
import torch as T
import torch.nn as _nn
import torch.optim as _optim

if T.cuda.is_available():
    _dev = T.device("cuda:0")
    # Touch CUDA to initialise the CUDA context
    _ = T.zeros(1, device=_dev)
    # Exercise nn and optim to force-load all lazy C extensions
    _dummy_net = _nn.Sequential(
        _nn.Linear(4, 4),
        _nn.LayerNorm(4),
        _nn.ReLU(),
    ).to(_dev)
    _dummy_opt = _optim.AdamW(_dummy_net.parameters(), lr=1e-3)
    # Forward + backward to initialise cuBLAS/cuDNN handles inside PyTorch
    _dummy_out = _dummy_net(T.zeros(1, 4, device=_dev))
    _dummy_out.sum().backward()
    _dummy_opt.step()
    del _dummy_net, _dummy_opt, _dummy_out, _
else:
    _dev = T.device("cpu")
    _dummy_net = _nn.Sequential(_nn.Linear(4, 4), _nn.LayerNorm(4), _nn.ReLU())
    _dummy_opt = _optim.AdamW(_dummy_net.parameters(), lr=1e-3)
    _dummy_out = _dummy_net(T.zeros(1, 4))
    _dummy_out.sum().backward()
    del _dummy_net, _dummy_opt, _dummy_out

# ── Step 2: Now it is safe to import TF — restrict it to CPU only ──────────
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass  # TF not installed or already configured
