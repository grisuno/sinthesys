# Polyglot Codebase Knowledge Graph

> Generated offline by **readmenator**. Supports C, C++, Python, Go, Rust, JS/TS, Java, C#, Shell, PHP, Dart, GDScript, Nim, ASM.
> No LLMs. No tokens. Pure static analysis. See more [here](https://github.com/grisuno/ReadMenator)

**Total Files Parsed:** 2 | **Total Symbols Extracted:** 17 | **Total Imports:** 9

## Structural Knowledge Map
```mermaid
graph TD
    classDef mod fill:#1e1e1e,stroke:#ff6666,stroke-width:2px,color:#fff;
    classDef cls fill:#2d2d2d,stroke:#4ec9b0,stroke-width:2px,color:#fff;
    classDef fn fill:#333,stroke:#dcdcaa,stroke-width:1px,color:#dcdcaa;
    classDef ext fill:#111,stroke:#666,stroke-dasharray:5 5,color:#aaa;
    app_py["app.py (py)"]
    class app_py mod;
    app_py_RealWorldEnvironment["RealWorldEnvironment"]
    class app_py_RealWorldEnvironment cls;
    app_py --> app_py_RealWorldEnvironment
    app_py_measure_spatial_richness["measure_spatial_richness"]
    class app_py_measure_spatial_richness fn;
    app_py --> app_py_measure_spatial_richness
    app_py_HomeostasisEngine["HomeostasisEngine"]
    class app_py_HomeostasisEngine cls;
    app_py --> app_py_HomeostasisEngine
    app_py_LiquidNeuron["LiquidNeuron"]
    class app_py_LiquidNeuron cls;
    app_py --> app_py_LiquidNeuron
    app_py_OrganismV8_Real["OrganismV8_Real"]
    class app_py_OrganismV8_Real cls;
    app_py --> app_py_OrganismV8_Real
    install_sh["install.sh (sh)"]
    class install_sh mod;
    ext_torch["torch"]
    class ext_torch ext;
    app_py -.->|imports| ext_torch
    ext_torch_nn["torch.nn"]
    class ext_torch_nn ext;
    app_py -.->|imports| ext_torch_nn
    ext_torch_nn_functional["torch.nn.functional"]
    class ext_torch_nn_functional ext;
    app_py -.->|imports| ext_torch_nn_functional
    ext_torch_optim["torch.optim"]
    class ext_torch_optim ext;
    app_py -.->|imports| ext_torch_optim
    ext_numpy["numpy"]
    class ext_numpy ext;
    app_py -.->|imports| ext_numpy
    ext_sklearn_datasets["sklearn.datasets"]
    class ext_sklearn_datasets ext;
    app_py -.->|imports| ext_sklearn_datasets
    ext_sklearn_model_selection["sklearn.model_selection"]
    class ext_sklearn_model_selection ext;
    app_py -.->|imports| ext_sklearn_model_selection
    ext_logging["logging"]
    class ext_logging ext;
    app_py -.->|imports| ext_logging
    ext_warnings["warnings"]
    class ext_warnings ext;
    app_py -.->|imports| ext_warnings
```

---

## Architecture Reference

### PY (1 files)

#### `app.py`
**Path:** `app.py`

**Classes:**
- `RealWorldEnvironment` (line 31) `class RealWorldEnvironment`
- `HomeostasisEngine` (line 80) `class HomeostasisEngine`
- `LiquidNeuron` (line 99) `class LiquidNeuron`
- `OrganismV8_Real` (line 136) `class OrganismV8_Real`

**Functions:**
- `measure_spatial_richness` (line 68) `def measure_spatial_richness(activations)`
- `run_real_world_challenge` (line 169) `def run_real_world_challenge()`
- `__init__` (line 32) `def __init__(self)`
- `get_batch` (line 50) `def get_batch(self, phase, batch_size)`
- `__init__` (line 81) `def __init__(self)`
- `decide` (line 85) `def decide(self, task_loss_val, richness_val, vn_entropy_val)`
- `__init__` (line 100) `def __init__(self, in_dim, out_dim)`
- `forward` (line 108) `def forward(self, x, plasticity_gate)`
- `consolidate_svd` (line 123) `def consolidate_svd(self, repair_strength)`
- `__init__` (line 137) `def __init__(self, d_in, d_hid, d_out)`
- `forward` (line 147) `def forward(self, x, plasticity_gate)`
- `get_structure_entropy` (line 155) `def get_structure_entropy(self)`
- `calc_ent` (line 157) `def calc_ent(W)`

### SH (1 files)

#### `install.sh`
**Path:** `install.sh`

*No symbols extracted*
