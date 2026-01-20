#!/bin/bash
set -euo pipefail

# Run heavy installs only once per node
DONEFILE="/tmp/hipool_install_done_${SLURM_JOB_ID:-$$}"

# Only rank 0 does the installation; others wait
if [[ "${SLURM_LOCALID:-0}" != "0" ]]; then
  # Wait until rank 0 finishes installation
  while [[ ! -f "${DONEFILE}" ]]; do
    sleep 1
  done
  exit 0
fi

echo ">>> [install] Python version:"
python -c "import sys; print(sys.version)"

echo ">>> [install] NumPy version BEFORE install:"
python - << 'PYCHECK' || true
try:
    import numpy as np
    print("NumPy:", np.__version__)
except Exception as e:
    print("NumPy not importable:", e)
PYCHECK

echo ">>> [install] Upgrading pip..."
python -m pip install --upgrade pip

BREAK_FLAG=""
if python -m pip --help | grep -q "break-system-packages"; then
  BREAK_FLAG="--break-system-packages"
fi

echo ">>> [install] Forcing NumPy 1.26.4 (to avoid NumPy 2.x binary issues)..."
python -m pip install --no-cache-dir --force-reinstall "numpy==1.26.4" ${BREAK_FLAG}

echo ">>> [install] NumPy version AFTER reinstall:"
python - << 'PYCHECK'
import numpy as np
print("NumPy:", np.__version__)
PYCHECK

echo ">>> [install] Removing problematic binary packages (soxr, pyarrow) if present..."
python -m pip uninstall -y soxr pyarrow ${BREAK_FLAG} || true

echo ">>> [install] Installing core scientific/audio stack..."
python -m pip install --no-cache-dir --upgrade-strategy only-if-needed \
  "scipy==1.12.0" \
  "pandas==2.2.2" \
  "scikit-learn" \
  "matplotlib==3.9.1" \
  "tqdm" \
  "librosa==0.10.1" \
  "soundfile==0.12.*" \
  "audiomentations==0.36.0" \
  "numba>=0.59,<0.61" \
  ${BREAK_FLAG}

echo ">>> [install] Installing training utilities..."
python -m pip install --no-cache-dir --upgrade-strategy only-if-needed \
  "einops==0.8.1" \
  "hydra-core==1.3.2" \
  "omegaconf==2.3.0" \
  "lightning==2.5.5" \
  "torchmetrics==1.8.2" \
  ${BREAK_FLAG}

echo ">>> [install] Installing torchaudio (this was missing)..."
# Do NOT pin a specific version here; let pip match the already-installed torch
python -m pip install --no-cache-dir --upgrade-strategy only-if-needed \
  torchaudio \
  ${BREAK_FLAG}

echo ">>> [install] Installing remaining WSSED dependencies from sed_torch.yaml..."
python -m pip install --no-cache-dir --upgrade-strategy only-if-needed \
  "appdirs==1.4.4" \
  "audioread==2.1.9" \
  "cffi==1.14.6" \
  "charset-normalizer==2.0.4" \
  "cycler==0.10.0" \
  "dcase-util==0.2.18" \
  "decorator==5.0.9" \
  "future==0.18.2" \
  "idna==3.2" \
  "jedi==0.18.0" \
  "joblib==1.0.1" \
  "kiwisolver==1.3.1" \
  "llvmlite==0.37.0" \
  "packaging==21.0" \
  "parso==0.8.2" \
  "pillow==8.3.1" \
  "pooch==1.5.1" \
  "pudb==2021.1" \
  "pycparser==2.20" \
  "pydot-ng==2.0.0" \
  "pygments==2.10.0" \
  "pyparsing==2.4.7" \
  "python-dateutil==2.8.2" \
  "python-magic==0.4.24" \
  "pytz==2021.1" \
  "pyyaml==5.4.1" \
  "requests==2.26.0" \
  "resampy==0.2.2" \
  "sed-eval==0.2.1" \
  "six==1.16.0" \
  "threadpoolctl==2.2.0" \
  "typing-extensions==3.10.0.0" \
  "urllib3==1.26.6" \
  "urwid==2.1.2" \
  "validators==0.18.2" \
  ${BREAK_FLAG}

echo ">>> [install] Cleaning up unwanted extras (if they exist)..."
python -m pip uninstall -y torch-audiomentations numpy-minmax ${BREAK_FLAG} || true

echo ">>> [install] Final sanity check (torch / torchaudio / numpy):"
python - << 'PYCHECK'
import importlib


def safe_import(name):
    try:
        m = importlib.import_module(name)
        print(f"{name}:", getattr(m, "__version__", "no __version__ attr"))
    except Exception as e:
        print(f"{name} import FAILED:", e)


safe_import("numpy")
safe_import("torch")
safe_import("torchaudio")
safe_import("pandas")
PYCHECK

touch "${DONEFILE}"
echo ">>> [install] Done."
