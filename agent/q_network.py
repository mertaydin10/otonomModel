"""
q_network.py — PyTorch Q-Network
DQL ajanının kullandığı sinir ağı mimarisi.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Deep Q-Network (DQN) — Dueling Architecture.

    Giriş: state_size boyutlu durum vektörü
    Çıkış: action_size adet Q değeri

    Katman mimarisi (Dueling DQN):
    ┌──────────────────────────┐
    │  Paylaşılan Özellik Ağı  │
    │  Dense(64) → Dense(64)   │
    └────────────┬─────────────┘
                 │
       ┌─────────┴──────────┐
       │                    │
    Value stream        Advantage stream
    Dense(32) → V(s)    Dense(32) → A(s,a)
       │                    │
       └─────────┬──────────┘
                 │
            Q(s,a) = V(s) + (A(s,a) - mean(A))
    """

    def __init__(self, state_size: int = 9, action_size: int = 4, hidden_size: int = 64):
        super().__init__()
        self.state_size  = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        # Paylaşılan özellik ağı
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Value stream: tek bir skaler değer V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Advantage stream: her aksiyon için avantaj A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, action_size),
        )

        # Ağırlıkları başlat
        self._init_weights()

    def _init_weights(self) -> None:
        """He (Kaiming) initialization — ReLU ağları için optimal."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        İleri geçiş.

        Args:
            x: (batch_size, state_size) tensor

        Returns:
            q_values: (batch_size, action_size) tensor
        """
        features  = self.feature(x)
        value     = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Dueling formülü: Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

    def get_action(self, state: torch.Tensor) -> int:
        """Açgözlü aksiyon seçimi (inference için)."""
        with torch.no_grad():
            q_values = self.forward(state.unsqueeze(0))
            return int(q_values.argmax(dim=1).item())
