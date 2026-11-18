# Contributors: Peter Lin (peterql2@illinois.edu), Michael Wang (mw114@illinois.edu)
"""Deep Cox Mixtures on SUPPORT2: example usage and ablation study.

Paper: Nagpal et al., "Deep Cox Mixtures for Survival Regression", MLHC 2021.
https://proceedings.mlr.press/v149/nagpal21a.html

Experimental setup
------------------
Data: PyHealth's ``Support2Dataset`` with the existing
``SurvivalTimeToEventSupport2`` task (8,873 patients with complete numeric
covariates, 27 features, time to death in days). Patients are split 70/10/20
with ``split_by_patient``; features are standardized with training-set
statistics only. ``--synthetic`` instead generates a two-group dataset whose
groups have crossing hazards, so the script runs without downloads.

Model: ``DeepCoxMixtures`` trained with approximate EM (``fit_em``), batch size
256, up to 50 epochs, early stopping on validation loss with patience 5.

Ablations (each varies one factor from the reference configuration
K=3, hidden_dims=(100,), lr=1e-3, dropout=0):

* number of groups K in {1, 2, 3, 4, 6}; K=1 is DeepSurv with a smoothed
  Breslow baseline,
* encoder: linear (``hidden_dims=()``), one hidden layer, two hidden layers,
* learning rate in {1e-3, 1e-4},
* dropout in {0, 0.25}.

Metrics (``pyhealth.metrics.survival.survival_metrics_fn``) on the test split:
Harrell's C-index of ``1 - S(t_50 | x)``, and cumulative/dynamic AUC and IPCW
Brier score at the 25th, 50th and 75th percentiles of training event times.
Each configuration is repeated over ``--seeds`` random seeds (mean +/- std).

Findings (SUPPORT2, 3 seeds; see the table printed by the script)
-----------------------------------------------------------------
==========================  ===============  ===============  ===============
Configuration               C-index          AUC@q25          Brier@q50
==========================  ===============  ===============  ===============
K=1 (DeepSurv)              0.592 +/- 0.009  0.580 +/- 0.013  0.226 +/- 0.009
K=2                         0.594 +/- 0.006  0.642 +/- 0.020  0.218 +/- 0.008
K=3 (reference)             0.594 +/- 0.003  0.619 +/- 0.025  0.220 +/- 0.008
K=4                         0.590 +/- 0.009  0.618 +/- 0.039  0.220 +/- 0.009
K=6                         0.593 +/- 0.006  0.615 +/- 0.032  0.218 +/- 0.009
linear encoder              0.576 +/- 0.010  0.586 +/- 0.024  0.222 +/- 0.005
two hidden layers           0.602 +/- 0.006  0.622 +/- 0.026  0.218 +/- 0.005
lr=1e-4                     0.545 +/- 0.028  0.555 +/- 0.025  0.227 +/- 0.005
dropout=0.25                0.582 +/- 0.009  0.586 +/- 0.016  0.222 +/- 0.004
==========================  ===============  ===============  ===============

* Mixtures (K >= 2) improve early-horizon discrimination over the single-group
  DeepSurv special case (AUC@q25 0.58 -> 0.62-0.64) and lower the Brier
  score, while Harrell's C-index stays flat; gains saturate by K=2-3.
* A nonlinear encoder helps: linear < one hidden layer < two hidden layers.
* lr=1e-4 underfits within 50 epochs, and dropout 0.25 slightly hurts.
* Absolute scores are lower than the paper's because this task keeps the
  27-feature DeepSurv benchmark representation of SUPPORT2 rather than the
  paper's full covariate set.

Usage::

    python examples/support2_survival_deep_cox_mixtures.py --synthetic
    python examples/support2_survival_deep_cox_mixtures.py --root /path/to/support2
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from pyhealth.datasets import (
    Support2Dataset,
    create_sample_dataset,
    get_dataloader,
    split_by_patient,
)
from pyhealth.metrics.survival import survival_metrics_fn
from pyhealth.models import DeepCoxMixtures
from pyhealth.tasks import SurvivalTimeToEventSupport2

REFERENCE = {"num_groups": 3, "hidden_dims": (100,), "lr": 1e-3, "dropout": 0.0}
ABLATIONS = {
    "num_groups": [1, 2, 3, 4, 6],
    "hidden_dims": [(), (100,), (100, 100)],
    "lr": [1e-3, 1e-4],
    "dropout": [0.0, 0.25],
}

Arrays = tuple[np.ndarray, np.ndarray, np.ndarray]


def synthetic_arrays(n: int = 1500, seed: int = 0) -> Arrays:
    """Two latent groups with opposite covariate effects (crossing hazards)."""
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, 6)).astype(np.float32)
    group = features[:, 1] > 0
    rate = np.where(group, np.exp(features[:, 0]), 0.2 * np.exp(-features[:, 0]))
    event_time = rng.exponential(1.0 / rate) + 0.01
    censor_time = rng.exponential(3.0, size=n) + 0.01
    duration = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(np.float32)
    return features, duration.astype(np.float32), event


def to_arrays(dataset) -> Arrays:
    """Stack a PyHealth survival dataset into feature, duration, event arrays."""
    rows = [dataset[i] for i in range(len(dataset))]
    features = np.stack([np.asarray(row["features"]) for row in rows])
    duration = np.array([float(row["duration"][0]) for row in rows])
    event = np.array([float(row["event"][0]) for row in rows])
    return features.astype(np.float32), duration, event


def support2_splits(root: str, seed: int) -> dict[str, Arrays]:
    """Load SUPPORT2 through PyHealth and split 70/10/20 by patient."""
    base = Support2Dataset(root=root, tables=["support2"])
    samples = base.set_task(SurvivalTimeToEventSupport2())
    train, val, test = split_by_patient(samples, [0.7, 0.1, 0.2], seed=seed)
    return {"train": to_arrays(train), "val": to_arrays(val), "test": to_arrays(test)}


def synthetic_splits(seed: int) -> dict[str, Arrays]:
    """Split the synthetic dataset 70/10/20."""
    features, duration, event = synthetic_arrays(seed=seed)
    order = np.random.default_rng(seed).permutation(len(duration))
    cut = np.cumsum([int(0.7 * len(order)), int(0.1 * len(order))])
    parts = np.split(order, cut)
    return {
        name: (features[idx], duration[idx], event[idx])
        for name, idx in zip(("train", "val", "test"), parts)
    }


def standardize(splits: dict[str, Arrays]) -> dict[str, Arrays]:
    """Scale features with training-set mean and standard deviation."""
    mean = splits["train"][0].mean(axis=0)
    scale = splits["train"][0].std(axis=0)
    scale[scale == 0] = 1.0
    return {
        name: ((x - mean) / scale, duration, event)
        for name, (x, duration, event) in splits.items()
    }


def sample_dataset(arrays: Arrays, name: str):
    """Wrap arrays as a PyHealth ``SampleDataset``."""
    features, duration, event = arrays
    samples = [
        {
            "patient_id": f"{name}-{i}",
            "features": features[i].tolist(),
            "duration": [float(duration[i])],
            "event": [float(event[i])],
        }
        for i in range(len(duration))
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"features": "tensor"},
        output_schema={"duration": "tensor", "event": "tensor"},
        dataset_name=name,
    )


def run_config(
    datasets: dict, splits: dict[str, Arrays], config: dict, epochs: int, seed: int
) -> dict[str, float]:
    """Train one configuration and return its test metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DeepCoxMixtures(
        datasets["train"],
        num_groups=config["num_groups"],
        hidden_dims=config["hidden_dims"],
        dropout=config["dropout"],
    )
    model.fit_em(
        get_dataloader(datasets["train"], batch_size=256, shuffle=True),
        get_dataloader(datasets["val"], batch_size=1024, shuffle=False),
        epochs=epochs,
        lr=config["lr"],
    )
    train_duration, train_event = splits["train"][1], splits["train"][2]
    horizons = np.quantile(train_duration[train_event == 1], [0.25, 0.5, 0.75])
    features, duration, event = splits["test"]
    survival = model.predict_survival(torch.as_tensor(features), model.time_grid)
    risk = model.predict_risk(torch.as_tensor(features), float(horizons[1]))
    metrics = survival_metrics_fn(
        duration,
        event,
        risk.numpy(),
        survival=survival.numpy(),
        time_grid=model.time_grid.numpy(),
        times=tuple(horizons),
    )
    renamed = {"c_index": metrics["c_index"]}
    for quantile, horizon in zip(("q25", "q50", "q75"), horizons):
        for name in ("auc", "brier"):
            renamed[f"{name}@{quantile}"] = metrics[f"{name}@{horizon:g}"]
    return renamed


def configurations() -> list[tuple[str, dict]]:
    """One-factor-at-a-time variations of the reference configuration."""
    seen, configs = set(), []
    for factor, values in ABLATIONS.items():
        for value in values:
            config = REFERENCE | {factor: value}
            key = tuple(sorted(config.items()))
            if key not in seen:
                seen.add(key)
                configs.append((f"{factor}={value}", config))
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", help="Directory containing support2.csv")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    if not args.synthetic and args.root is None:
        parser.error("pass --root /path/to/support2 or --synthetic")

    rows = []
    for seed in range(args.seeds):
        splits = synthetic_splits(seed) if args.synthetic else support2_splits(
            args.root, seed
        )
        splits = standardize(splits)
        datasets = {name: sample_dataset(arr, name) for name, arr in splits.items()}
        for label, config in configurations():
            metrics = run_config(datasets, splits, config, args.epochs, seed)
            rows.append({"config": label, "seed": seed} | metrics)
            print(f"seed {seed} {label}: c-index {metrics['c_index']:.3f}")

    table = pd.DataFrame(rows).drop(columns="seed").groupby("config", sort=False)
    summary = table.mean().round(3).astype(str) + " +/- " + table.std().round(3).astype(
        str
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
