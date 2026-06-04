# MLP model for credit risk classification (SentinelAI project)
# Team: Sai Yaswanth, Lavanya, Sirichandana - Secure & Private AI, GSU
#
# Important note about GroupNorm: Opacus (the DP-SGD library we're using
# in Phase 2) breaks with BatchNorm because BatchNorm computes stats
# across the whole batch. DP-SGD needs per-sample gradients, so we use
# GroupNorm(num_groups=1) instead - it does the same job but per-sample.

import torch
import torch.nn as nn


class CreditRiskMLP(nn.Module):
    """
    4-layer feedforward network for binary loan default prediction.
    Architecture: Input -> 256 -> 128 -> 64 -> 2 (Default / No Default)

    We kept it straightforward - dropout for regularization, GroupNorm
    so we can drop in Opacus later without changing the architecture.
    """

    def __init__(self, n_features: int, dropout_rate: float = 0.3):
        super(CreditRiskMLP, self).__init__()

        self.network = nn.Sequential(
            # layer 1
            nn.Linear(n_features, 256),
            nn.GroupNorm(num_groups=1, num_channels=256),  # NOT BatchNorm - see header note
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # layer 2
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # layer 3
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # output
            nn.Linear(64, 2),
        )

        self._init_weights()

    def _init_weights(self):
        # Xavier init to avoid vanishing gradients in the deeper layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        # we use this internally for evaluation but the API never
        # returns raw probabilities (that would help model extraction)
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model(n_features: int, dropout_rate: float = 0.3) -> CreditRiskMLP:
    model = CreditRiskMLP(n_features=n_features, dropout_rate=dropout_rate)
    print(f"[Model] CreditRiskMLP | features={n_features} | trainable params={model.count_parameters():,}")
    return model
