"""Phase 8 smoke test script."""
import sys, os
sys.path.insert(0, r'D:\College\Semesters\6th Sem\mini-project')
import numpy as np
from src.environment.counterfactual_simulator import (
    CounterfactualSimulator, DISCOUNT_BINS, STATE_DIM, INPUT_DIM
)

print('=== PHASE 8 SMOKE TEST ===')
print(f'STATE_DIM     = {STATE_DIM}')
print(f'INPUT_DIM     = {INPUT_DIM}')
print(f'DISCOUNT_BINS = {DISCOUNT_BINS}')

# Load simulator
print('\n[1] Loading CounterfactualSimulator...')
sim = CounterfactualSimulator()
print('    OK -- predictor loaded and frozen')
print(f'    input_dim = {sim.input_dim}')
assert sim.input_dim == 112, f'input_dim mismatch: {sim.input_dim}'

print('\n[2] Summary:')
print(sim.summary())

# Load a real state row
state_file = r'D:\College\Semesters\6th Sem\mini-project\data\processed\state\state_train.npy'
state_arr = np.load(state_file)
state = state_arr[0].astype('float32')
print(f'\n[3] Sample state shape={state.shape}, finite={bool(np.all(np.isfinite(state)))}')

# Single predict
print('\n[4] Single predictions:')
for d in DISCOUNT_BINS:
    p = sim.predict(state, d)
    flag = '  [out-of-support]' if d == 0 else ''
    print(f'    discount={d:2d}%  predicted_growth={p:.4f}{flag}')

# predict_all_discounts
print('\n[5] predict_all_discounts():')
all_preds = sim.predict_all_discounts(state)
assert len(all_preds) == 9, f'Expected 9 keys, got {len(all_preds)}'
vals = [all_preds[k] for k in sorted(all_preds.keys())]
print(f'    Keys: {sorted(all_preds.keys())}')
print(f'    Values: {[round(v,4) for v in vals]}')
var = np.var(vals)
print(f'    Variance across discounts: {var:.6f}')
assert np.std(vals) > 0, 'Predictions are constant -- model not responding to discount'

# Batch predict
print('\n[6] predict_batch():')
states_batch = np.tile(state, (9, 1))
discounts_batch = np.array(DISCOUNT_BINS, dtype='float32')
batch_out = sim.predict_batch(states_batch, discounts_batch)
assert batch_out.shape == (9,), f'Bad batch shape: {batch_out.shape}'
print(f'    Output shape: {batch_out.shape}, finite: {bool(np.all(np.isfinite(batch_out)))}')

single_out = np.array([sim.predict(state, d) for d in DISCOUNT_BINS])
np.testing.assert_allclose(batch_out, single_out, rtol=1e-5)
print('    Batch == Single: PASS')

# Validation guards
print('\n[7] Validation guards:')
try:
    sim.predict(np.ones(80, dtype='float32'), 10)
    print('    ERROR: bad state dim should have raised ValueError!')
except ValueError:
    print('    bad state dim -> ValueError: OK')

try:
    sim.predict(state, 7)
    print('    ERROR: bad discount should have raised ValueError!')
except ValueError:
    print('    bad discount -> ValueError: OK')

nan_state = np.full(STATE_DIM, float('nan'), dtype='float32')
try:
    sim.predict(nan_state, 10)
    print('    ERROR: NaN state should have raised ValueError!')
except ValueError:
    print('    NaN state -> ValueError: OK')

print('\n=== SMOKE TEST PASSED ===')
