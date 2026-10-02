# Transfer Adapter Stage

Files:
- `src/transfer/transfer_adapter.py`
- `scripts/train_transfer_adapter.py`
- `tests/test_transfer_adapter.py`

The base paper defines pseudo-state alignment as a learnable mapping from
the 121-D target state to the 4-D NeoRL source state, followed by action-space
compatible policy distillation.

The current local implementation has an important data limitation: the public
Amazon data and NeoRL data do not contain genuine paired target/source states.
Therefore this stage does **not** fabricate pairs. Instead, it uses the
observed Amazon discount as the behavioral signal for adapting the frozen
NeoRL source policy into the target domain.

This should be described as a local transfer adaptation necessitated by
available data, not as an exact reproduction of the paper's paired-state
alignment objective.

Run from the project root:

```powershell
python scripts\train_transfer_adapter.py --epochs 20 --batch-size 512
```

Expected output:
`models/transfer/adapter/best_adapter.pth`
