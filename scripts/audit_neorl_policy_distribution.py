from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transfer.neorl_source_policy import (
    NeoRLSourcePolicy,
    NeoRLSourcePolicyConfig,
)


DEVICE = torch.device("cpu")


def load_policy():
    model = NeoRLSourcePolicy(
        NeoRLSourcePolicyConfig()
    ).to(DEVICE)

    checkpoint = (
        PROJECT_ROOT
        / "models/transfer/neorl_source_policy/best_model.pth"
    )

    payload = torch.load(
        checkpoint,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        payload.get(
            "model_state_dict",
            payload,
        )
    )

    model.eval()

    return model


def describe(name, x):
    x = np.asarray(x).reshape(-1)

    print(f"\n{name}")
    print("-" * 60)
    print(f"mean : {x.mean():.8f}")
    print(f"std  : {x.std():.8f}")
    print(f"min  : {x.min():.8f}")
    print(f"p01  : {np.percentile(x, 1):.8f}")
    print(f"p05  : {np.percentile(x, 5):.8f}")
    print(f"p50  : {np.percentile(x, 50):.8f}")
    print(f"p95  : {np.percentile(x, 95):.8f}")
    print(f"p99  : {np.percentile(x, 99):.8f}")
    print(f"max  : {x.max():.8f}")


def main():
    print("=" * 80)
    print("NEORL SOURCE POLICY DISTRIBUTION AUDIT")
    print("=" * 80)

    policy = load_policy()

    print("\nSource policy class:")
    print(type(policy))

    print("\nforward() signature:")
    print(inspect.signature(policy.forward))

    if hasattr(policy, "sample_action"):
        print("\nsample_action() signature:")
        print(inspect.signature(policy.sample_action))
    else:
        print("\nsample_action() does NOT exist.")

    if hasattr(policy, "deterministic_action"):
        print("\ndeterministic_action() signature:")
        print(inspect.signature(policy.deterministic_action))

    # Use representative pseudo-states covering the range
    # actually produced by the current adapter.
    rng = np.random.default_rng(42)

    states = np.column_stack(
        [
            rng.normal(16.0, 7.0, 4096),
            rng.normal(0.49, 0.15, 4096),
            rng.normal(8.5, 2.6, 4096),
            rng.normal(2.96, 0.10, 4096),
        ]
    ).astype(np.float32)

    x = torch.from_numpy(states).to(DEVICE)

    with torch.no_grad():

        output = policy.forward(x)

        print("\n" + "=" * 80)
        print("RAW FORWARD OUTPUT")
        print("=" * 80)

        if isinstance(output, tuple):
            print(
                f"Forward returned tuple of length "
                f"{len(output)}"
            )

            for i, value in enumerate(output):
                if torch.is_tensor(value):
                    print(
                        f"Output[{i}] shape: "
                        f"{tuple(value.shape)}"
                    )

                    describe(
                        f"Output[{i}] dimension 0",
                        value[:, 0].cpu().numpy(),
                    )

                    describe(
                        f"Output[{i}] dimension 1",
                        value[:, 1].cpu().numpy(),
                    )
                else:
                    print(
                        f"Output[{i}] type: "
                        f"{type(value)}"
                    )

            first = output[0]
            second = output[1]

        else:
            print(
                f"Forward returned: "
                f"{type(output)}"
            )

            if not torch.is_tensor(output):
                raise RuntimeError(
                    "Unexpected forward() output."
                )

            first = output

            if output.ndim != 2 or output.shape[1] < 2:
                raise RuntimeError(
                    "Forward output does not have "
                    "two action dimensions."
                )

            second = None

        print("\n" + "=" * 80)
        print("INTERPRETATION CHECK")
        print("=" * 80)

        if second is not None:

            second_np = second.cpu().numpy()

            describe(
                "Second forward output, raw",
                second_np[:, 1],
            )

            raw = second[:, 1]

            describe(
                "exp(second output)",
                torch.exp(raw).cpu().numpy(),
            )

            describe(
                "softplus(second output)",
                torch.nn.functional.softplus(raw)
                .cpu()
                .numpy(),
            )

            describe(
                "clamped second output",
                raw.clamp(
                    -5.0,
                    5.0,
                ).cpu().numpy(),
            )

        # If sample_action exists, directly inspect actual samples.
        if hasattr(policy, "sample_action"):

            print("\n" + "=" * 80)
            print("ACTUAL SOURCE-POLICY SAMPLES")
            print("=" * 80)

            samples = []

            for _ in range(100):
                sampled = policy.sample_action(x)

                if isinstance(sampled, tuple):
                    sampled = sampled[0]

                if torch.is_tensor(sampled):
                    samples.append(
                        sampled.detach().cpu().numpy()
                    )

            samples = np.concatenate(
                samples,
                axis=0,
            )

            print(
                f"Collected samples: "
                f"{len(samples)}"
            )

            describe(
                "Sampled action dimension 0",
                samples[:, 0],
            )

            describe(
                "Sampled action dimension 1",
                samples[:, 1],
            )

            # Interpret the sampled second action using
            # the NeoRL dataset normalization.
            factor = samples[:, 1] * 5.0

            discount = (
                1.0 - factor
            ) * 100.0

            describe(
                "Sampled discount "
                "(action * 5 interpretation)",
                discount,
            )

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()