# Soft Policy Distillation: NeoRL -> ItaNet

This stage is an experimental behavioral-transfer branch.

## Purpose

The previous distillation stage converted the frozen NeoRL source policy
into hard discount labels using deterministic action selection. Because the
current 111D -> 4D adapter is concentrated around a narrow source-policy
behavior, the resulting ItaNet policy selected 10% for every product.

This branch preserves the source policy's second-action probability
distribution using Monte-Carlo sampling and distills it into the supported
8-action ItaNet discount space:

5, 10, 15, 20, 25, 30, 35, 40%.

## Important limitation

This is not an exact implementation of the paper's paired-state transfer
equation. Amazon product states and NeoRL states are different state spaces,
so this is a behavioral adaptation experiment.

The existing hard-label distillation checkpoint is intentionally not
overwritten.

## Run order

1. Tests:

```powershell
python -m pytest tests\test_soft_policy_distillation.py -q
```

2. Smoke training:

```powershell
python scripts\train_soft_policy_distillation.py --epochs 2 --batch-size 512 --samples 32
```

3. Inspect the printed mean target distribution.

Do not run a long training job until the target distribution is inspected.
