# AGENTS.md

## Cursor Cloud specific instructions

This is a **research ML codebase** (SIGGRAPH Asia paper "Data-driven Interior Plan Generation for Residential Buildings"), not a web application. There are no services, Docker containers, databases, or CI/CD pipelines.

### Architecture

Four sequential neural network stages generate interior floor plans from a building boundary:

1. **Living** (`train/Living/`) – predicts living room location
2. **Continue** (`train/Continue/`) – decides whether to add more rooms
3. **Location** (`train/Location/`) – predicts next room type and position
4. **Wall** (`train/Wall/`) – generates interior walls

Synthesis pipeline is in `synth/`. Post-processing vectorization is in `vectorization.py`.

### Dependencies

Install via pip (CPU-only PyTorch is sufficient for development without GPU):

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install torchnet fire tqdm numpy scipy
```

No `requirements.txt` exists in the repo. Pillow is installed as a torch/torchvision dependency.

### Git submodules (`third_party/`)

Two submodules must be initialized after cloning:

```
git submodule update --init --recursive
```

- **RPLAN-Toolbox** (`third_party/RPLAN-Toolbox/`): Toolbox for loading/processing RPLAN dataset. Extra deps: `scikit-image`, `matplotlib`, `shapely`. Optional: MATLAB (for alignment), `faiss` (for clustering).
- **Graph2plan** (`third_party/Graph2plan/`): Graph-based floorplan generation (SIGGRAPH 2020). Contains Network, Interface (Django web app), and PostProcess modules. Extra deps: `pytorch-ignite`, `django`, `opencv-python`, `pandas`, `shapely`.

### Key caveats

- **No GPU in Cloud Agent VMs**: All training scripts and `synth/synth.py` call `.cuda()`. For CPU-only development, model instantiation and forward passes work, but actual training/synthesis requires wrapping with device-aware code or a GPU environment.
- **Python version**: Code was written for Python 3.6. `time.clock()` (removed in Python 3.8) is used in `synth/synth.py` (lines 264, 284, 288) and `vectorization.py` (lines 755, 778, 782). These will raise `AttributeError` on Python 3.8+. Replace with `time.perf_counter()` if you need to run those main blocks.
- **No test suite**: There are no automated tests, linting config, or CI setup. Validation is done by running the pipeline end-to-end.
- **No dataset included**: The RPLAN dataset must be downloaded separately. Training requires `dataset/train` and `dataset/val` directories with PNG images, preprocessed via `python write_pickle.py`.
- **Module imports**: Training scripts (e.g., `train/Living/train_living.py`) must be run from their own directory with the repo root on `sys.path` for `utils` and `models` imports to resolve.
- **synth_input**: 10 sample boundary PNG files are provided in `synth/synth_input/` for synthesis testing (requires trained models in `synth/trained_model/`).
