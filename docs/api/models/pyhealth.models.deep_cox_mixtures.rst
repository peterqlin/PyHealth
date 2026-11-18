pyhealth.models.DeepCoxMixtures
===============================

Overview
--------

Deep Cox Mixtures (DCM) models right-censored survival as a mixture of ``K``
Cox proportional-hazards models, following Nagpal, Yadlowsky, Rostamzadeh and
Heller, `Deep Cox Mixtures for Survival Regression
<https://proceedings.mlr.press/v149/nagpal21a.html>`_ (MLHC 2021). A shared
encoder produces gate probabilities ``pi_k(x)`` and bounded log-hazard ratios
``f_k(x)``. Each latent group has its own spline-smoothed Breslow baseline
``S0_k(t)``, and the predicted survival is
``S(t | x) = sum_k pi_k(x) * S0_k(t) ** exp(f_k(x))``. Proportional hazards
therefore only has to hold within a group.

Inputs and outputs
------------------

Use a sample dataset with ``features: tensor`` as the input schema and
``duration: tensor, event: tensor`` as the output schema, for example
:class:`~pyhealth.datasets.Support2Dataset` with
:class:`~pyhealth.tasks.SurvivalTimeToEventSupport2`. Event is ``1`` for an
observed event and ``0`` for censoring.

``forward`` returns the M-step loss (per-group Cox partial likelihood on group
assignments sampled from the E-step posterior, plus the gate cross-entropy),
``y_true`` with columns duration and event, ``y_prob`` with survival on
``time_grid`` (``[batch, len(time_grid)]``), and the gate probabilities,
log-hazard ratios and posteriors. ``logit`` concatenates gate logits and
log-hazard ratios. ``mode`` is ``None``: score predictions with
:mod:`pyhealth.metrics.survival`.

Training
--------

Baseline curves are re-estimated with the Breslow estimator rather than by
gradient descent, so train with :meth:`~DeepCoxMixtures.fit_em`, which
alternates baseline refits on the full training set with E-step / M-step
minibatch updates and early-stops on validation loss::

    model = DeepCoxMixtures(train_dataset, num_groups=3, hidden_dims=(100,))
    model.fit_em(train_loader, val_loader, epochs=50, lr=1e-3)
    survival = model.predict_survival(features, times)

``num_groups=1`` gives DeepSurv with a smoothed Breslow baseline, and
``hidden_dims=()`` gives a linear encoder.

Example and ablation
--------------------

``examples/support2_survival_deep_cox_mixtures.py`` varies the number of
groups, the encoder depth, the learning rate and dropout on SUPPORT2 over
several seeds. Run it with ``--synthetic`` to use generated data.

API
---

.. autoclass:: pyhealth.models.DeepCoxMixtures
   :members:
   :show-inheritance:
