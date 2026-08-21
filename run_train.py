import gpu_setup  # noqa: F401 — must be first to prevent TF/PyTorch CUDA conflict
from peptide_optimization.framework import Framework

if __name__ == "__main__":

    framework = Framework()
    framework.train()
    