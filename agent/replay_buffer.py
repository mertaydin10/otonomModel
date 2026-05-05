"""
replay_buffer.py — Experience Replay Buffer
Prioritized Experience Replay (PER) ile birlikte standart
uniform sampling destekli bellek modülü.
"""
import numpy as np
import random
from collections import deque
from typing import Tuple, List


class ReplayBuffer:
    """
    Uniform Experience Replay Buffer.

    Deneyimleri (s, a, r, s', done) saklar ve
    mini-batch örneklemesi yapar.
    """

    def __init__(self, capacity: int = 10_000):
        """
        Args:
            capacity: Maksimum bellek kapasitesi
        """
        self.buffer   = deque(maxlen=capacity)
        self.capacity = capacity

    def add(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        """Yeni deneyim ekle."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Rastgele mini-batch örnekle.

        Returns:
            (states, actions, rewards, next_states, dones) — numpy array'ler
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Yeterli deneyim birikti mi?"""
        return len(self.buffer) >= batch_size


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer (PER).

    TD hatası büyük olan deneyimleri daha sık örnekler.
    Daha verimli öğrenme sağlar.

    Referans: Schaul et al., 2015 — "Prioritized Experience Replay"
    """

    def __init__(
        self,
        capacity:  int   = 10_000,
        alpha:     float = 0.6,   # önceliklendirme kuvveti
        beta_start:float = 0.4,   # IS düzeltme başlangıcı
        beta_end:  float = 1.0,   # IS düzeltme sonu
        beta_steps:int   = 100_000,
    ):
        self.capacity   = capacity
        self.alpha      = alpha
        self.beta_start = beta_start
        self.beta_end   = beta_end
        self.beta_steps = beta_steps
        self.step_count = 0

        self.buffer     = []
        self.priorities  = np.zeros(capacity, dtype=np.float32)
        self.pos        = 0
        self.max_prio   = 1.0

    @property
    def beta(self) -> float:
        """Lineer beta artışı."""
        frac = min(1.0, self.step_count / self.beta_steps)
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    def add(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = self.max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, List[int], np.ndarray]:
        """
        Öncelikli örnekleme.

        Returns:
            (batch_data, indices, weights) — IS ağırlıkları ile
        """
        self.step_count += 1
        n       = len(self.buffer)
        prios   = self.priorities[:n]
        probs   = prios ** self.alpha
        probs  /= probs.sum()

        indices = np.random.choice(n, batch_size, replace=False, p=probs)
        samples = [self.buffer[i] for i in indices]

        # Importance Sampling ağırlıkları
        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)

        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        ), list(indices), weights.astype(np.float32)

    def update_priorities(
        self, indices: List[int], td_errors: np.ndarray
    ) -> None:
        """TD hatalarına göre öncelikleri güncelle."""
        for idx, err in zip(indices, td_errors):
            prio = (abs(err) + 1e-6) ** self.alpha
            self.priorities[idx] = prio
            self.max_prio        = max(self.max_prio, prio)

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size
