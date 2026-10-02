import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.environment.counterfactual_simulator import (
    CounterfactualSimulator,
)


STATE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "state"
    / "state.npy"
)


DISCOUNTS = [0, 5, 10, 15, 20, 25, 30, 35, 40]


def main():

    print("=" * 65)
    print("PHASE 11E.1 - COUNTERFACTUAL DISCOUNT ANALYSIS")
    print("=" * 65)

    states = np.load(STATE_PATH)

    simulator = CounterfactualSimulator()

    print("\nState matrix:", states.shape)
    print("Simulator state dimension:", simulator.state_dim)

    # Select deterministic real products.
    indices = [
        0,
        1000,
        5000,
        10000,
        15000,
    ]

    print("\nPredicted sales-growth across discount levels:\n")

    for index in indices:

        state = states[index]

        predictions = simulator.predict_all_discounts(
            state
        )

        print(f"Product/state index: {index}")

        for discount in DISCOUNTS:

            prediction = float(
                predictions[discount]
            )

            print(
                f"  Discount {discount:2d}%"
                f" -> prediction {prediction:12.6f}"
            )

        values = np.array(
            [
                predictions[d]
                for d in DISCOUNTS
            ],
            dtype=np.float64,
        )

        print(
            "  Min prediction:",
            values.min()
        )

        print(
            "  Max prediction:",
            values.max()
        )

        print(
            "  Range:",
            values.max() - values.min()
        )

        print(
            "  Finite:",
            np.isfinite(values).all()
        )

        print()

    print("=" * 65)
    print("COUNTERFACTUAL ANALYSIS COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()