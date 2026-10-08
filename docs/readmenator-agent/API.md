# API

## app.py
- `RealWorldEnvironment.__init__` (method) `app.py:32` `def __init__(self)`
- `RealWorldEnvironment.get_batch` (method) `app.py:50` `def get_batch(self, phase, batch_size)`
- `RealWorldEnvironment.measure_spatial_richness` (method) `app.py:68` `def measure_spatial_richness(activations)`
- `HomeostasisEngine.__init__` (method) `app.py:81` `def __init__(self)`
- `HomeostasisEngine.decide` (method) `app.py:85` `def decide(self, task_loss_val, richness_val, vn_entropy_val)`
- `LiquidNeuron.__init__` (method) `app.py:100` `def __init__(self, in_dim, out_dim)`
- `LiquidNeuron.forward` (method) `app.py:108` `def forward(self, x, plasticity_gate)`
- `LiquidNeuron.consolidate_svd` (method) `app.py:123` `def consolidate_svd(self, repair_strength)`
- `OrganismV8_Real.__init__` (method) `app.py:137` `def __init__(self, d_in, d_hid, d_out)`
- `OrganismV8_Real.forward` (method) `app.py:147` `def forward(self, x, plasticity_gate)`
- `OrganismV8_Real.get_structure_entropy` (method) `app.py:155` `def get_structure_entropy(self)`
- `OrganismV8_Real.calc_ent` (method) `app.py:157` `def calc_ent(W)`
- `OrganismV8_Real.run_real_world_challenge` (method) `app.py:169` `def run_real_world_challenge()`
