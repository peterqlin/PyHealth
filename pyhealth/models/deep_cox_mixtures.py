# Contributors: Peter Lin (peterql2@illinois.edu), Michael Wang (mw114@illinois.edu)
"""Deep Cox Mixtures for right-censored survival regression.

This module implements Deep Cox Mixtures (DCM) from Nagpal, Yadlowsky,
Rostamzadeh and Heller, "Deep Cox Mixtures for Survival Regression"
(MLHC 2021). A shared encoder maps covariates to a soft assignment over ``K``
latent groups and to one log-hazard ratio per group. Each group has its own
nonparametric baseline survival curve, estimated with the Breslow estimator
and smoothed with a spline, so proportional hazards only has to hold within a
group. Training is approximate EM; see :meth:`DeepCoxMixtures.fit_em`.

Paper: https://proceedings.mlr.press/v149/nagpal21a.html
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Sequence

import numpy as np
import torch
from scipy.interpolate import UnivariateSpline
from torch import Tensor, nn

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel
from pyhealth.processors import LabelProcessor

__all__ = ["DeepCoxMixtures"]

_EPS = 1e-10


def _value_tensor(value: Tensor | tuple[Tensor, ...]) -> Tensor:
    """Extract the primary value emitted by a PyHealth processor."""
    if isinstance(value, tuple):
        return value[0]
    return value


def _infer_input_dim(dataset: SampleDataset, feature_key: str) -> int:
    """Infer the dense feature width from the first processed sample."""
    value = _value_tensor(dataset[0][feature_key])
    tensor = torch.as_tensor(value)
    if tensor.ndim != 1:
        raise ValueError(
            "DeepCoxMixtures expects each sample's feature field to be a 1D "
            "dense numeric vector"
        )
    return int(tensor.shape[0])


def _cox_partial_nll(log_risk: Tensor, duration: Tensor, event: Tensor) -> Tensor:
    """Summed negative Cox partial log-likelihood with Breslow ties.

    Args:
        log_risk: Log-hazard ratios with shape ``[n]``.
        duration: Observed times with shape ``[n]``.
        event: Event indicators with shape ``[n]``.

    Returns:
        Scalar negative log partial likelihood, summed over events.
    """
    if log_risk.numel() == 0 or event.sum() == 0:
        return log_risk.sum() * 0.0
    order = torch.argsort(duration, descending=True)
    log_risk, duration, event = log_risk[order], duration[order], event[order]
    log_cum = torch.logcumsumexp(log_risk, dim=0)
    # Under Breslow ties every subject tied at t is in the risk set at t, so
    # each tie group uses the largest cumulative sum in that group.
    _, tie_group = torch.unique_consecutive(duration, return_inverse=True)
    group_max = torch.full(
        (int(tie_group.max()) + 1,), -torch.inf, device=log_risk.device
    )
    group_max = group_max.scatter_reduce(0, tie_group, log_cum, reduce="amax")
    log_denominator = group_max[tie_group]
    return -((log_risk - log_denominator) * event).sum()


def _interpolate(times: Tensor, grid: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
    """Evaluate piecewise-linear curves and their slopes.

    Args:
        times: Query times with any shape ``S``.
        grid: Increasing knot times with shape ``[G]``.
        values: Curve values at the knots with shape ``[K, G]``.

    Returns:
        Tuple ``(value, slope)``, each with shape ``S + [K]``. Times beyond the
        grid take the boundary value and zero slope.
    """
    clamped = times.clamp(grid[0], grid[-1])
    index = torch.searchsorted(grid, clamped, right=True).clamp(1, grid.numel() - 1)
    left, right = grid[index - 1], grid[index]
    width = (right - left).clamp_min(_EPS)
    value_left, value_right = values[:, index - 1], values[:, index]
    slope = (value_right - value_left) / width
    value = value_left + slope * (clamped - left)
    outside = (times > grid[-1]) | (times < grid[0])
    slope = torch.where(outside, torch.zeros_like(slope), slope)
    return value.movedim(0, -1), slope.movedim(0, -1)


class DeepCoxMixtures(BaseModel):
    """Mixture of Cox models with neural gating and per-group baselines.

    For covariates ``x`` the model predicts gate probabilities
    ``pi_k(x)`` and log-hazard ratios ``f_k(x) = gamma * tanh(w_k^T phi(x))``
    for ``k = 1..K``. With per-group baseline survival ``S0_k(t)``, the
    predicted survival is ``S(t | x) = sum_k pi_k(x) * S0_k(t) ** exp(f_k(x))``.
    With ``num_groups=1`` and an MLP encoder the model reduces to DeepSurv
    with a smoothed Breslow baseline; with ``hidden_dims=()`` and one group,
    to a linear Cox model.

    ``forward`` returns the M-step objective: the Cox partial likelihood of
    each group, evaluated on group assignments sampled from the E-step
    posterior, plus the gate cross-entropy against that posterior. Baseline
    curves are not learned by gradient descent; :meth:`fit_baselines` refits
    them. :meth:`fit_em` runs the full alternating procedure.

    Args:
        dataset: Fitted PyHealth ``SampleDataset`` with one dense feature
            vector input and ``duration`` / ``event`` tensor outputs.
        feature_key: Input field holding the dense feature vector.
        duration_key: Output field holding positive observed times.
        event_key: Output field holding ``1`` for an observed event and ``0``
            for right censoring.
        input_dim: Feature width. Inferred from the first sample if omitted.
        num_groups: Number of latent groups ``K``.
        hidden_dims: Hidden layer widths of the encoder ``phi``. An empty
            tuple gives a linear encoder.
        dropout: Dropout probability after each hidden layer.
        gamma: Bound on the absolute log-hazard ratio.
        smoothing_factor: Smoothing factor ``s`` of the baseline splines.
        time_grid: Times at which ``y_prob`` reports survival. If omitted, a
            grid over the training follow-up range is set by the first call
            to :meth:`fit_baselines`.
        num_time_points: Length of the default ``time_grid``.
        num_baseline_points: Number of knots used to store each baseline.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {"patient_id": "p0", "features": [0.1, 1.0],
        ...      "duration": [3.0], "event": [1.0]},
        ...     {"patient_id": "p1", "features": [-0.4, 0.2],
        ...      "duration": [5.0], "event": [0.0]},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"features": "tensor"},
        ...     output_schema={"duration": "tensor", "event": "tensor"},
        ... )
        >>> model = DeepCoxMixtures(dataset, num_groups=3, hidden_dims=(16,))
        >>> model.num_groups
        3

    References:
        Nagpal, C., Yadlowsky, S., Rostamzadeh, N., and Heller, K. Deep Cox
        Mixtures for Survival Regression. Machine Learning for Healthcare, 2021.
    """

    @property
    def label_processors(self) -> dict[str, LabelProcessor]:
        """None: survival outputs are scored by the model, not as a label."""
        return {}

    def __init__(
        self,
        dataset: SampleDataset,
        feature_key: str = "features",
        duration_key: str = "duration",
        event_key: str = "event",
        input_dim: int | None = None,
        num_groups: int = 3,
        hidden_dims: Sequence[int] = (100,),
        dropout: float = 0.0,
        gamma: float = 10.0,
        smoothing_factor: float = 1e-4,
        time_grid: Tensor | None = None,
        num_time_points: int = 100,
        num_baseline_points: int = 500,
    ) -> None:
        super().__init__(dataset)
        if feature_key not in dataset.input_schema:
            raise ValueError(
                f"feature_key {feature_key!r} is not in dataset.input_schema"
            )
        for key in (duration_key, event_key):
            if key not in dataset.output_schema:
                raise ValueError(f"output key {key!r} is not in dataset.output_schema")
        if num_groups < 1:
            raise ValueError("num_groups must be at least 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if gamma <= 0:
            raise ValueError("gamma must be positive")

        self.feature_key = feature_key
        self.duration_key = duration_key
        self.event_key = event_key
        self.num_groups = int(num_groups)
        self.hidden_dims = tuple(int(width) for width in hidden_dims)
        self.gamma = float(gamma)
        self.smoothing_factor = float(smoothing_factor)
        self.input_dim = int(
            input_dim if input_dim is not None
            else _infer_input_dim(dataset, feature_key)
        )

        layers: list[nn.Module] = []
        width = self.input_dim
        for hidden in self.hidden_dims:
            layers.extend([nn.Linear(width, hidden), nn.ReLU6(), nn.Dropout(dropout)])
            width = hidden
        self.encoder = nn.Sequential(*layers)
        self.gate = nn.Linear(width, self.num_groups, bias=False)
        self.expert = nn.Linear(width, self.num_groups, bias=False)

        self._user_time_grid = time_grid is not None
        if time_grid is None:
            time_grid = torch.zeros(num_time_points)
        self.register_buffer(
            "time_grid", torch.as_tensor(time_grid, dtype=torch.float32).reshape(-1)
        )
        self.register_buffer("baseline_times", torch.zeros(num_baseline_points))
        self.register_buffer(
            "baseline_survival", torch.ones(self.num_groups, num_baseline_points)
        )
        self.register_buffer("baselines_fitted", torch.tensor(False))

    def _network(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Return gate logits and bounded log-hazard ratios, each ``[B, K]``."""
        hidden = self.encoder(features)
        log_hazard_ratios = self.gamma * torch.tanh(self.expert(hidden))
        return self.gate(hidden), log_hazard_ratios

    def _posteriors(
        self,
        log_gate: Tensor,
        log_hazard_ratios: Tensor,
        duration: Tensor,
        event: Tensor,
    ) -> Tensor:
        """E-step posterior ``q(z = k | x, t, delta)`` with shape ``[B, K]``."""
        if not bool(self.baselines_fitted):
            return torch.full_like(log_gate, 1.0 / self.num_groups)
        survival0, slope0 = _interpolate(
            duration, self.baseline_times, self.baseline_survival
        )
        log_survival0 = survival0.clamp_min(_EPS).log()
        risk = log_hazard_ratios.exp()
        log_survival = risk * log_survival0
        log_density = (
            log_hazard_ratios
            + (risk - 1.0) * log_survival0
            + (-slope0).clamp_min(_EPS).log()
        )
        event = event.unsqueeze(-1)
        log_joint = log_gate + event * log_density + (1.0 - event) * log_survival
        return torch.softmax(log_joint, dim=-1)

    def _parse_batch(
        self, kwargs: dict[str, Tensor | tuple[Tensor, ...]]
    ) -> tuple[Tensor, Tensor, Tensor]:
        missing = [
            key
            for key in (self.feature_key, self.duration_key, self.event_key)
            if key not in kwargs
        ]
        if missing:
            raise KeyError(f"Missing required batch fields: {missing}")
        features = torch.as_tensor(
            _value_tensor(kwargs[self.feature_key]),
            dtype=torch.float32,
            device=self.device,
        )
        duration = torch.as_tensor(
            _value_tensor(kwargs[self.duration_key]),
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1)
        event = torch.as_tensor(
            _value_tensor(kwargs[self.event_key]),
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1)
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                "features must have shape [batch, input_dim]; "
                f"received {tuple(features.shape)}"
            )
        return features, duration, event

    def forward(self, **kwargs: Tensor | tuple[Tensor, ...]) -> dict[str, Tensor]:
        """Compute the M-step loss and survival predictions for one batch.

        In training mode group assignments are sampled from the posterior; in
        evaluation mode the most probable group is used.

        Args:
            **kwargs: Batch fields from the PyHealth dataloader, including
                ``feature_key``, ``duration_key`` and ``event_key``.

        Returns:
            Dictionary with ``loss``, ``logit`` (gate logits followed by
            log-hazard ratios, ``[B, 2K]``), ``y_prob`` (survival on
            ``time_grid``, ``[B, T]``; NaN until baselines are fitted),
            ``y_true`` (duration and event, ``[B, 2]``), ``gate_probs``,
            ``log_hazard_ratios`` and ``posteriors`` (each ``[B, K]``).
        """
        features, duration, event = self._parse_batch(kwargs)
        gate_logits, log_hazard_ratios = self._network(features)
        log_gate = torch.log_softmax(gate_logits, dim=-1)
        with torch.no_grad():
            posteriors = self._posteriors(
                log_gate, log_hazard_ratios, duration, event
            )
            if self.training:
                assignment = torch.multinomial(posteriors, 1).squeeze(-1)
            else:
                assignment = posteriors.argmax(dim=-1)

        cox_nll = sum(
            _cox_partial_nll(
                log_hazard_ratios[assignment == k, k],
                duration[assignment == k],
                event[assignment == k],
            )
            for k in range(self.num_groups)
        )
        cox_loss = cox_nll / event.sum().clamp_min(1.0)
        gate_loss = -(posteriors * log_gate).sum(dim=-1).mean()

        if bool(self.baselines_fitted):
            y_prob = self._mixture_survival(log_gate, log_hazard_ratios, self.time_grid)
        else:
            y_prob = torch.full(
                (features.shape[0], self.time_grid.numel()),
                torch.nan,
                device=self.device,
            )
        return {
            "loss": cox_loss + gate_loss,
            "logit": torch.cat([gate_logits, log_hazard_ratios], dim=-1),
            "y_prob": y_prob,
            "y_true": torch.stack([duration, event], dim=-1),
            "gate_probs": log_gate.exp(),
            "log_hazard_ratios": log_hazard_ratios,
            "posteriors": posteriors,
        }

    def _mixture_survival(
        self, log_gate: Tensor, log_hazard_ratios: Tensor, times: Tensor
    ) -> Tensor:
        """Mixture survival ``[B, T]`` from network outputs."""
        survival0, _ = _interpolate(
            times.reshape(-1), self.baseline_times, self.baseline_survival
        )
        log_survival0 = survival0.clamp_min(_EPS).log()
        risk = log_hazard_ratios.exp()
        log_component = risk.unsqueeze(1) * log_survival0.unsqueeze(0)
        survival = torch.logsumexp(log_gate.unsqueeze(1) + log_component, dim=-1)
        return survival.exp().clamp(0.0, 1.0)

    @torch.no_grad()
    def e_step(self, features: Tensor, duration: Tensor, event: Tensor) -> Tensor:
        """Compute posterior group probabilities.

        ``q(z = k | x, t, delta)`` is proportional to
        ``pi_k(x) * f_k(t | x) ** delta * S_k(t | x) ** (1 - delta)``, where
        ``f_k`` is the group density. Posteriors are uniform until baselines
        have been fitted.

        Args:
            features: Dense features ``[B, input_dim]``.
            duration: Observed times ``[B]``.
            event: Event indicators ``[B]``.

        Returns:
            Posterior probabilities ``[B, K]`` whose rows sum to one.
        """
        features, duration, event = self._parse_batch(
            {
                self.feature_key: features,
                self.duration_key: duration,
                self.event_key: event,
            }
        )
        gate_logits, log_hazard_ratios = self._network(features)
        return self._posteriors(
            torch.log_softmax(gate_logits, dim=-1), log_hazard_ratios, duration, event
        )

    @torch.no_grad()
    def fit_baselines(self, features: Tensor, duration: Tensor, event: Tensor) -> None:
        """Refit each group's smoothed Breslow baseline survival.

        Group assignments are sampled from the current E-step posterior. For
        group ``k``, the Breslow cumulative hazard is
        ``H0_k(t) = sum_{t_j <= t, delta_j = 1} 1 / sum_{l: t_l >= t_j}
        exp(f_k(x_l))`` over that group's members; ``S0_k = exp(-H0_k)`` is
        smoothed with a linear ``UnivariateSpline`` and stored on a fixed grid.

        Args:
            features: Dense training features ``[N, input_dim]``.
            duration: Training times ``[N]``.
            event: Training event indicators ``[N]``.
        """
        previous_modes = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            features, duration, event = self._parse_batch(
                {
                    self.feature_key: features,
                    self.duration_key: duration,
                    self.event_key: event,
                }
            )
            gate_logits, log_hazard_ratios = self._network(features)
            posteriors = self._posteriors(
                torch.log_softmax(gate_logits, dim=-1),
                log_hazard_ratios,
                duration,
                event,
            )
            assignment = torch.multinomial(posteriors, 1).squeeze(-1).cpu().numpy()
        finally:
            for module, training in previous_modes:
                module.training = training

        times = duration.cpu().numpy().astype(np.float64)
        events = event.cpu().numpy().astype(bool)
        risks = log_hazard_ratios.exp().cpu().numpy().astype(np.float64)
        grid = np.linspace(0.0, times.max(), self.baseline_times.numel())
        survival = np.ones((self.num_groups, grid.size))
        for k in range(self.num_groups):
            member = assignment == k
            survival[k] = self._breslow_on_grid(
                times[member], events[member], risks[member, k], grid
            )

        self.baseline_times.copy_(torch.as_tensor(grid, dtype=torch.float32))
        self.baseline_survival.copy_(torch.as_tensor(survival, dtype=torch.float32))
        self.baselines_fitted.fill_(True)
        if not self._user_time_grid:
            self.time_grid.copy_(
                torch.linspace(
                    times.max() / self.time_grid.numel(),
                    times.max(),
                    self.time_grid.numel(),
                )
            )

    def _breslow_on_grid(
        self,
        times: np.ndarray,
        events: np.ndarray,
        risks: np.ndarray,
        grid: np.ndarray,
    ) -> np.ndarray:
        """Smoothed Breslow baseline survival for one group, evaluated on grid."""
        event_times = np.unique(times[events])
        if event_times.size == 0:
            return np.ones_like(grid)
        at_risk = np.array([risks[times >= t].sum() for t in event_times])
        deaths = np.array([np.sum(events & (times == t)) for t in event_times])
        cumulative_hazard = np.cumsum(deaths / np.maximum(at_risk, _EPS))
        knots = np.concatenate([[0.0], event_times])
        survival = np.concatenate([[1.0], np.exp(-cumulative_hazard)])
        smoothed = np.interp(grid, knots, survival)
        if knots.size > 2:
            spline = UnivariateSpline(
                knots, survival, k=1, s=self.smoothing_factor, ext=3
            )
            fitted = spline(grid)
            # FITPACK can return NaN when a group has very few members.
            if np.isfinite(fitted).all():
                smoothed = fitted
        smoothed = np.minimum.accumulate(np.clip(smoothed, _EPS, 1.0))
        return smoothed

    def _collect(
        self, loader: Iterable[dict[str, Tensor | tuple[Tensor, ...]]]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Concatenate every batch of a dataloader into full tensors."""
        parts = [self._parse_batch(batch) for batch in loader]
        return tuple(torch.cat(column) for column in zip(*parts))

    @torch.no_grad()
    def _mean_loss(
        self, loader: Iterable[dict[str, Tensor | tuple[Tensor, ...]]]
    ) -> float:
        """Average evaluation-mode loss over a dataloader."""
        previous_modes = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            losses = [float(self(**batch)["loss"]) for batch in loader]
        finally:
            for module, training in previous_modes:
                module.training = training
        return float(np.mean(losses))

    def fit_em(
        self,
        train_loader: Iterable[dict[str, Tensor | tuple[Tensor, ...]]],
        val_loader: Iterable[dict[str, Tensor | tuple[Tensor, ...]]] | None = None,
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        patience: int = 5,
        verbose: bool = False,
    ) -> dict[str, list[float]]:
        """Train with approximate (hard, stochastic) EM.

        Each epoch first refits every group's baseline on the full training
        set (sampling assignments from the current posterior), then takes one
        Adam step per minibatch on the M-step loss returned by ``forward``,
        which itself runs the E-step and samples hard assignments. When a
        validation loader is given, the loss is evaluated after refitting the
        baselines, and the parameters and baselines with the lowest
        validation loss are restored at the end.

        Args:
            train_loader: Dataloader yielding training batches.
            val_loader: Optional dataloader for early stopping.
            epochs: Maximum number of epochs.
            lr: Adam learning rate.
            weight_decay: Adam weight decay.
            patience: Epochs without validation improvement before stopping.
            verbose: Print the losses after every epoch.

        Returns:
            History with ``train_loss`` and, if validating, ``val_loss`` lists.
        """
        features, duration, event = self._collect(train_loader)
        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_loss, best_state, stale = np.inf, None, 0
        for epoch in range(epochs):
            self.fit_baselines(features, duration, event)
            self.train()
            batch_losses = []
            for batch in train_loader:
                optimizer.zero_grad()
                loss = self(**batch)["loss"]
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            history["train_loss"].append(float(np.mean(batch_losses)))
            if val_loader is None:
                continue
            self.fit_baselines(features, duration, event)
            val_loss = self._mean_loss(val_loader)
            history["val_loss"].append(val_loss)
            if verbose:
                print(
                    f"epoch {epoch}: train {history['train_loss'][-1]:.4f} "
                    f"val {val_loss:.4f}"
                )
            if val_loss < best_loss:
                best_loss, stale = val_loss, 0
                best_state = copy.deepcopy(self.state_dict())
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is not None:
            self.load_state_dict(best_state)
        else:
            self.fit_baselines(features, duration, event)
        self.eval()
        return history

    @torch.no_grad()
    def predict_survival(self, features: Tensor, times: Tensor) -> Tensor:
        """Predict mixture survival probabilities.

        Args:
            features: Dense features ``[B, input_dim]``.
            times: Query times ``[T]`` in the training time units.

        Returns:
            Survival probabilities ``[B, T]``.

        Raises:
            RuntimeError: If baselines have not been fitted.
        """
        if not bool(self.baselines_fitted):
            raise RuntimeError("Call fit_baselines or fit_em before predicting")
        features = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        times = torch.as_tensor(times, dtype=torch.float32, device=self.device)
        previous_modes = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            gate_logits, log_hazard_ratios = self._network(features)
            return self._mixture_survival(
                torch.log_softmax(gate_logits, dim=-1), log_hazard_ratios, times
            )
        finally:
            for module, training in previous_modes:
                module.training = training

    def predict_risk(self, features: Tensor, horizon: float) -> Tensor:
        """Predict the event probability ``1 - S(horizon | x)``.

        Args:
            features: Dense features ``[B, input_dim]``.
            horizon: Time at which to evaluate risk.

        Returns:
            Risk scores ``[B]``; higher means earlier expected event.
        """
        times = torch.tensor([horizon], dtype=torch.float32)
        return 1.0 - self.predict_survival(features, times)[:, 0]
