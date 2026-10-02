from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import neorl

from src.transfer.neorl_source_policy import (
    NeoRLSourcePolicy,
    NeoRLSourcePolicyConfig,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_neorl_dataset():
    env = neorl.make("sp")

    train, val = env.get_dataset()

    return train, val


def prepare_dataset(dataset):
    observations = np.asarray(
        dataset["obs"],
        dtype=np.float32,
    )

    actions = np.asarray(
        dataset["action"],
        dtype=np.float32,
    )

    if observations.ndim != 2 or observations.shape[1] != 4:
        raise ValueError(
            f"Expected observations shape (N,4), "
            f"got {observations.shape}"
        )

    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError(
            f"Expected actions shape (N,2), "
            f"got {actions.shape}"
        )

    return observations, actions


def gaussian_nll(
    policy,
    states,
    actions,
):
    log_probability = policy.log_prob(
        states,
        actions,
    )

    return -log_probability.mean()


def evaluate(
    policy,
    loader,
    device,
):
    policy.eval()

    total_loss = 0.0
    total_samples = 0

    action_error_sum = 0.0

    with torch.no_grad():
        for states, actions in loader:
            states = states.to(device)
            actions = actions.to(device)

            loss = gaussian_nll(
                policy,
                states,
                actions,
            )

            predicted = policy.deterministic_action(
                states
            )

            batch_size = states.shape[0]

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            action_error_sum += (
                torch.abs(predicted - actions)
                .mean()
                .item()
                * batch_size
            )

    return {
        "nll": total_loss / total_samples,
        "mean_absolute_action_error": (
            action_error_sum / total_samples
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=8e-4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/transfer/neorl_source_policy",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cpu")

    print("=" * 70)
    print("NeoRL Source Policy Training")
    print("=" * 70)

    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")

    train_data, val_data = load_neorl_dataset()

    train_states, train_actions = prepare_dataset(
        train_data
    )

    val_states, val_actions = prepare_dataset(
        val_data
    )

    print()
    print("Dataset:")
    print(f"Train states:  {train_states.shape}")
    print(f"Train actions: {train_actions.shape}")
    print(f"Val states:    {val_states.shape}")
    print(f"Val actions:   {val_actions.shape}")

    train_dataset = TensorDataset(
        torch.from_numpy(train_states),
        torch.from_numpy(train_actions),
    )

    val_dataset = TensorDataset(
        torch.from_numpy(val_states),
        torch.from_numpy(val_actions),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    config = NeoRLSourcePolicyConfig(
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    policy = NeoRLSourcePolicy(config).to(device)

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=args.learning_rate,
        weight_decay=config.weight_decay,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_nll = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        policy.train()

        running_loss = 0.0
        samples = 0

        for states, actions in train_loader:
            states = states.to(device)
            actions = actions.to(device)

            optimizer.zero_grad()

            loss = gaussian_nll(
                policy,
                states,
                actions,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            batch_size = states.shape[0]

            running_loss += loss.item() * batch_size
            samples += batch_size

        train_nll = running_loss / samples

        val_metrics = evaluate(
            policy,
            val_loader,
            device,
        )

        record = {
            "epoch": epoch,
            "train_nll": train_nll,
            **val_metrics,
        }

        history.append(record)

        print(
            f"Epoch {epoch:03d} | "
            f"Train NLL {train_nll:.6f} | "
            f"Val NLL {val_metrics['nll']:.6f} | "
            f"Val MAE {val_metrics['mean_absolute_action_error']:.6f}"
        )

        if val_metrics["nll"] < best_val_nll:
            best_val_nll = val_metrics["nll"]

            torch.save(
                {
                    "model_state_dict": policy.state_dict(),
                    "config": vars(config),
                    "best_val_nll": best_val_nll,
                },
                output_dir / "best_model.pth",
            )

    with open(
        output_dir / "training_history.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            history,
            f,
            indent=2,
        )

    with open(
        output_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "seed": args.seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "state_dim": 4,
                "action_dim": 2,
                "device": str(device),
            },
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("Training complete")
    print("=" * 70)
    print(f"Best validation NLL: {best_val_nll:.6f}")
    print(f"Checkpoint: {output_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()