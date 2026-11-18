# Contributors: Peter Lin (peterql2@illinois.edu), Michael Wang (mw114@illinois.edu)
"""Fast synthetic tests for :class:`pyhealth.models.DeepCoxMixtures`."""

import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import DeepCoxMixtures

FEATURES = [
    [0.0, 1.0, -0.5],
    [1.0, 0.2, 0.4],
    [-0.5, 0.7, 0.1],
    [0.3, -0.4, 1.0],
    [0.8, -1.0, -0.3],
]
DURATIONS = [0.8, 1.5, 2.2, 2.2, 3.1]
EVENTS = [1.0, 0.0, 1.0, 1.0, 0.0]


def _make_dataset():
    samples = [
        {
            "patient_id": f"p{i}",
            "features": FEATURES[i],
            "duration": [DURATIONS[i]],
            "event": [EVENTS[i]],
        }
        for i in range(len(FEATURES))
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"features": "tensor"},
        output_schema={"duration": "tensor", "event": "tensor"},
        dataset_name="deep_cox_mixtures_test",
    )


def _tensors():
    return (
        torch.tensor(FEATURES),
        torch.tensor(DURATIONS),
        torch.tensor(EVENTS),
    )


class TestDeepCoxMixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = _make_dataset()
        cls.batch = next(iter(get_dataloader(cls.dataset, batch_size=5)))

    def setUp(self):
        torch.manual_seed(0)
        self.model = DeepCoxMixtures(
            self.dataset,
            num_groups=3,
            hidden_dims=(8,),
            time_grid=torch.linspace(0.5, 3.0, 6),
            num_baseline_points=20,
        )

    def test_instantiation(self):
        self.assertEqual(self.model.input_dim, 3)
        self.assertEqual(self.model.num_groups, 3)
        self.assertIsNone(self.model.mode)
        self.assertFalse(bool(self.model.baselines_fitted))

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            DeepCoxMixtures(self.dataset, feature_key="missing")
        with self.assertRaises(ValueError):
            DeepCoxMixtures(self.dataset, duration_key="missing")
        with self.assertRaises(ValueError):
            DeepCoxMixtures(self.dataset, num_groups=0)

    def test_forward_output_shapes(self):
        output = self.model(**self.batch)
        self.assertEqual(output["loss"].ndim, 0)
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertEqual(output["logit"].shape, (5, 6))
        self.assertEqual(output["y_prob"].shape, (5, 6))
        self.assertEqual(output["y_true"].shape, (5, 2))
        for key in ("gate_probs", "log_hazard_ratios", "posteriors"):
            self.assertEqual(output[key].shape, (5, 3))
        self.assertTrue(torch.isnan(output["y_prob"]).all())

    def test_gradients_reach_all_parameters(self):
        self.model.train()
        self.model(**self.batch)["loss"].backward()
        for name, parameter in self.model.named_parameters():
            if name == "_dummy_param":
                continue
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_log_hazard_ratio_bound(self):
        model = DeepCoxMixtures(self.dataset, hidden_dims=(), gamma=0.5)
        output = model(**self.batch)
        self.assertTrue((output["log_hazard_ratios"].abs() <= 0.5).all())

    def test_posteriors_sum_to_one(self):
        features, duration, event = _tensors()
        uniform = self.model.e_step(features, duration, event)
        self.assertTrue(torch.allclose(uniform, torch.full((5, 3), 1 / 3)))
        self.model.fit_baselines(features, duration, event)
        posteriors = self.model.e_step(features, duration, event)
        self.assertTrue(torch.allclose(posteriors.sum(-1), torch.ones(5)))

    def test_survival_predictions_are_valid(self):
        features, duration, event = _tensors()
        with self.assertRaises(RuntimeError):
            self.model.predict_survival(features, torch.tensor([1.0]))
        self.model.fit_baselines(features, duration, event)
        survival = self.model.predict_survival(features, torch.linspace(0, 4, 9))
        self.assertEqual(survival.shape, (5, 9))
        self.assertTrue(((survival >= 0) & (survival <= 1)).all())
        self.assertTrue((survival.diff(dim=-1) <= 1e-6).all())
        risk = self.model.predict_risk(features, 2.0)
        self.assertEqual(risk.shape, (5,))
        output = self.model(**self.batch)
        self.assertFalse(torch.isnan(output["y_prob"]).any())

    def test_breslow_falls_back_when_spline_is_not_finite(self):
        grid = np.linspace(0.0, 3.0, 7)
        times = np.array([0.5, 1.0, 2.0])
        events = np.array([True, True, True])
        risks = np.ones(3)
        with patch(
            "pyhealth.models.deep_cox_mixtures.UnivariateSpline",
            return_value=lambda x: np.full_like(x, np.nan),
        ):
            survival = self.model._breslow_on_grid(times, events, risks, grid)
        self.assertTrue(np.isfinite(survival).all())
        self.assertAlmostEqual(survival[0], 1.0)

    def test_single_group_linear_variant(self):
        model = DeepCoxMixtures(self.dataset, num_groups=1, hidden_dims=())
        output = model(**self.batch)
        self.assertTrue(torch.allclose(output["gate_probs"], torch.ones(5, 1)))
        output["loss"].backward()
        self.assertIsNotNone(model.expert.weight.grad)

    def test_fit_em_runs(self):
        loader = get_dataloader(self.dataset, batch_size=5, shuffle=False)
        history = self.model.fit_em(loader, loader, epochs=2, lr=1e-2)
        self.assertEqual(len(history["train_loss"]), 2)
        self.assertEqual(len(history["val_loss"]), 2)
        self.assertTrue(bool(self.model.baselines_fitted))

    def test_state_dict_round_trip(self):
        features, duration, event = _tensors()
        self.model.fit_baselines(features, duration, event)
        expected = self.model.predict_survival(features, torch.tensor([1.0, 2.0]))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "dcm.pt")
            torch.save(self.model.state_dict(), path)
            restored = DeepCoxMixtures(
                self.dataset,
                num_groups=3,
                hidden_dims=(8,),
                time_grid=torch.linspace(0.5, 3.0, 6),
                num_baseline_points=20,
            )
            restored.load_state_dict(torch.load(path))
        actual = restored.predict_survival(features, torch.tensor([1.0, 2.0]))
        self.assertTrue(torch.allclose(actual, expected))


if __name__ == "__main__":
    unittest.main()
