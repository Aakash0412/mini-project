# Transfer Policy Distillation

This stage distills behavior from the frozen NeoRL source policy into a
9-action target pricing policy over the 111-D ItaNet-111 state.

It intentionally preserves the earlier adapter checkpoint as an experiment.
The current implementation is a local behavioral-transfer adaptation, not an
exact reproduction of the paper's paired-state Eq. (21), because the public
Amazon and NeoRL datasets do not provide natural one-to-one cross-domain state
pairs.

Pipeline:

111-D Amazon state
-> existing 111-D to 4-D adapter
-> frozen NeoRL source policy
-> mapped 5-40% discount target
-> 9-action target policy

0% remains a paper-defined target action but has no NeoRL source counterpart.

Commands:

python -m pytest tests\test_policy_distillation.py -q

python scripts\train_policy_distillation.py --epochs 2 --batch-size 512

After the smoke test passes, run:

python scripts\train_policy_distillation.py --epochs 20 --batch-size 512
