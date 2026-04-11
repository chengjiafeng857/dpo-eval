"""Minimal math helpers for Arena-Hard v2.0 score reporting."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


class _BTModel(torch.nn.Module):
    def __init__(self, num_components: int) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(
            torch.nn.init.constant_(torch.empty(num_components), 0.5)
        )

    def forward(self) -> torch.Tensor:
        return self.logits


def fit_pairwise_model(
    features: torch.Tensor,
    outcomes: torch.Tensor,
    indices: torch.Tensor | None = None,
    *,
    lr: float = 0.1,
    tol: float = 1e-9,
    max_epochs: int = 50,
) -> torch.Tensor:
    if indices is not None:
        features = features[indices]
        outcomes = outcomes[indices]

    model = _BTModel(num_components=features.shape[1])
    optimizer = optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=max_epochs,
        tolerance_grad=tol,
        tolerance_change=tol,
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = features @ model()
        loss = F.binary_cross_entropy_with_logits(
            logits,
            outcomes.float(),
            reduction="sum",
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return model().detach()


def bootstrap_pairwise_model(
    features: torch.Tensor,
    outcomes: torch.Tensor,
    *,
    num_round: int = 100,
) -> torch.Tensor:
    if features.numel() == 0:
        raise ValueError("Cannot bootstrap an empty feature matrix.")
    count = features.shape[0]
    coefs = []
    for _ in tqdm(range(num_round), disable=num_round <= 1):
        indices = torch.randint(low=0, high=count, size=(count,))
        coefs.append(fit_pairwise_model(features, outcomes, indices))
    return torch.stack(coefs)


def one_hot_encode(items: Sequence[str], *, baseline: str) -> tuple[torch.Tensor, list[str]]:
    unique_items = sorted(set(items) | {baseline})
    item_to_index = {item: idx for idx, item in enumerate(unique_items)}

    rows = []
    for item in items:
        row = [0.0] * len(unique_items)
        row[item_to_index[item]] = 1.0
        row[item_to_index[baseline]] = -1.0
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32), unique_items


def to_winrate_probabilities(
    coefs: torch.Tensor,
    models: Sequence[str],
    *,
    baseline_model: str,
) -> torch.Tensor:
    baseline_idx = list(models).index(baseline_model)
    exp_coefs = torch.exp(coefs)
    probs = torch.zeros(coefs.shape[0], coefs.shape[1], dtype=torch.float32)
    for idx in range(len(models)):
        probs[:, idx] = exp_coefs[:, idx] / (
            exp_coefs[:, idx] + exp_coefs[:, baseline_idx]
        )
    probs[:, baseline_idx] = 0.5
    return probs

