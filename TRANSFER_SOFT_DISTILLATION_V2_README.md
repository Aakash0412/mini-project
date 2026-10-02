# Corrected Soft Policy Distillation V2

This is a separate experimental branch. It does not overwrite the previous
hard-label or first soft-distillation checkpoints.

## Correction

The previous experiment accidentally aggregated sampled actions into a
batch-level target and then discretized too early. This V2 implementation:

1. generates source-policy samples independently for every Amazon state;
2. converts each continuous source discount into an 8-class soft distribution;
3. averages only the Monte-Carlo samples belonging to that same state;
4. trains the 111D target policy using KL divergence.

Target actions:

5, 10, 15, 20, 25, 30, 35, 40%.

## Commands

Run tests:

```powershell
python -m pytest tests\test_soft_policy_distillation_v2.py -q
```

Then run only the smoke experiment:

```powershell
python scripts\train_soft_policy_distillation_v2.py --epochs 2 --batch-size 512 --samples 32
```

Do not run a long experiment until the printed target distribution is
inspected.

## Interpretation

This remains behavioral transfer through the existing 111D -> 4D adapter.
It is not an exact paired-state implementation of the paper's transfer
equation because Amazon product states and NeoRL states are different
spaces.
